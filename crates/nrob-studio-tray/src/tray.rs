//! The Windows notification-area icon: a hidden window receives the icon's
//! clicks, a popup menu drives the studio, balloons announce finished media.
//!
//! All the FFI of this crate is here. std has no Win32 UI, so each call goes
//! through `windows-sys`; every `unsafe` block says what makes it sound.

use nrob::json::Json;
use nrob_studio::Running;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows_sys::Win32::UI::HiDpi::{SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Shell::{
    Shell_NotifyIconW, NIF_ICON, NIF_INFO, NIF_MESSAGE, NIF_TIP, NIIF_INFO, NIIF_WARNING, NIM_ADD, NIM_DELETE, NIM_MODIFY,
    NOTIFYICONDATAW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const ICON: &[u8] = include_bytes!("../assets/nrob-studio.ico");
const CALLBACK: u32 = WM_APP + 1;
const TIMER: usize = 1;

const OPEN: usize = 1;
const TOGGLE_LLM: usize = 2;
const OUTPUTS: usize = 3;
const AUTOSTART: usize = 4;
const BROWSER: usize = 5;
const QUIT: usize = 9;

/// What the window procedure needs; set once before the window exists.
struct Tray {
    running: Running,
    /// Finished media job ids already announced.
    announced: Mutex<Vec<String>>,
    icon: AtomicIsize,
}

static TRAY: OnceLock<Tray> = OnceLock::new();
/// "TaskbarCreated": Explorer restarted, the icon must be added again.
static TASKBAR_CREATED: AtomicIsize = AtomicIsize::new(0);

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

/// Copy `s` into a fixed UTF-16 field, truncated, always terminated.
fn fill(field: &mut [u16], s: &str) {
    let mut i = 0;
    for u in s.encode_utf16().take(field.len() - 1) {
        field[i] = u;
        i += 1;
    }
    field[i] = 0;
}

pub fn message_box(text: &str) {
    let (text, title) = (wide(text), wide("NROB Studio"));
    // SAFETY: both pointers are NUL-terminated UTF-16 buffers that outlive the
    // call; a null owner window is allowed.
    unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR) };
}

/// The image in an .ico file that best fits `size` pixels: the smallest at
/// least that big, else the largest. As `(offset, length)` into the file.
fn ico_image(ico: &[u8], size: i32) -> Option<(usize, usize)> {
    let u16_at = |i: usize| Some(u16::from_le_bytes([*ico.get(i)?, *ico.get(i + 1)?]));
    let u32_at = |i: usize| Some(u32::from_le_bytes([*ico.get(i)?, *ico.get(i + 1)?, *ico.get(i + 2)?, *ico.get(i + 3)?]));
    let count = u16_at(4)? as usize;
    let mut entries = Vec::new();
    for i in 0..count {
        let e = 6 + i * 16;
        let width = match *ico.get(e)? {
            0 => 256,
            w => w as i32,
        };
        let (len, offset) = (u32_at(e + 8)? as usize, u32_at(e + 12)? as usize);
        if offset.checked_add(len).is_some_and(|end| end <= ico.len()) {
            entries.push((width, offset, len));
        }
    }
    let fits = entries.iter().filter(|e| e.0 >= size).min_by_key(|e| e.0);
    let (_, offset, len) = *fits.or_else(|| entries.iter().max_by_key(|e| e.0))?;
    Some((offset, len))
}

fn load_icon() -> HICON {
    // SAFETY: GetSystemMetrics takes no pointers.
    let size = unsafe { GetSystemMetrics(SM_CXSMICON) }.max(16);
    // LookupIconIdFromDirectoryEx expects a resource's group directory (14-byte
    // entries), not a file's (16-byte), so the file is read here.
    let Some((offset, len)) = ico_image(ICON, size) else { return std::ptr::null_mut() };
    let image = &ICON[offset..offset + len];
    // SAFETY: `image` is one complete icon image inside ICON (bounds checked
    // above); CreateIconFromResourceEx copies what it needs.
    unsafe { CreateIconFromResourceEx(image.as_ptr(), image.len() as u32, 1, 0x0003_0000, size, size, LR_DEFAULTCOLOR) }
}

fn notify(hwnd: HWND, message: u32, balloon: Option<(&str, &str, bool)>) {
    let Some(tray) = TRAY.get() else { return };
    let mut data = NOTIFYICONDATAW {
        cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
        hWnd: hwnd,
        uID: 1,
        uFlags: NIF_ICON | NIF_MESSAGE | NIF_TIP,
        uCallbackMessage: CALLBACK,
        hIcon: tray.icon.load(Ordering::Relaxed) as HICON,
        ..Default::default()
    };
    fill(&mut data.szTip, &tooltip(tray));
    if let Some((title, text, warn)) = balloon {
        data.uFlags |= NIF_INFO;
        fill(&mut data.szInfoTitle, title);
        fill(&mut data.szInfo, text);
        data.dwInfoFlags = if warn { NIIF_WARNING } else { NIIF_INFO };
    }
    // SAFETY: `data` is a fully initialised NOTIFYICONDATAW with its true size in
    // cbSize, and it lives across the call.
    unsafe { Shell_NotifyIconW(message, &data) };
}

fn llm_state(tray: &Tray) -> (String, bool) {
    let status = tray.running.studio.llm.status();
    let state = status.get("state").and_then(Json::as_str).unwrap_or("stopped").to_string();
    let running = matches!(state.as_str(), "ready" | "starting");
    (state, running)
}

fn tooltip(tray: &Tray) -> String {
    let (state, _) = llm_state(tray);
    let media = match tray.running.studio.media.running() {
        Some((kind, progress)) => format!("{} {progress:.0}%", kind.name()),
        None => "media idle".into(),
    };
    format!("NROB Studio: LLM {state}, {media}")
}

fn open_ui(tray: &Tray, mode: &str) {
    if mode == "app" {
        show_ui(&tray.running.ui_url);
    } else {
        nrob_studio::open_ui(&tray.running.ui_url, mode);
    }
}

/// Sharp icons and menus on high-DPI screens (Windows 10 1703+; harmless where
/// unsupported). Must run before any window exists.
pub fn dpi_aware() {
    // SAFETY: takes a constant context value, no pointers; failure is ignored.
    unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
}

/// When the last UI window was launched: a new app window takes a few seconds
/// to appear (and to take its title), so clicks meanwhile must not open more.
static LAST_LAUNCH: Mutex<Option<Instant>> = Mutex::new(None);

fn utf16_text(buf: &[u16], len: i32) -> String {
    String::from_utf16_lossy(&buf[..len.clamp(0, buf.len() as i32) as usize])
}

/// The studio's app window, if one is open: a top-level Chromium window (Edge
/// or Chrome `--app`) titled with the page title, or with the UI's address
/// while the page is still loading. The tray's own hidden window has another
/// class, and browser tabs carry the browser's name in their title.
fn find_ui_window(ui_url: &str) -> Option<HWND> {
    struct Search {
        address: String,
        found: HWND,
    }
    unsafe extern "system" fn visit(hwnd: HWND, lparam: LPARAM) -> windows_sys::core::BOOL {
        // SAFETY: `lparam` is the `&mut Search` passed to EnumWindows below, which
        // outlives the enumeration; the buffers are sized as declared.
        unsafe {
            let search = &mut *(lparam as *mut Search);
            if IsWindowVisible(hwnd) == 0 {
                return 1;
            }
            let mut class = [0u16; 64];
            let n = GetClassNameW(hwnd, class.as_mut_ptr(), class.len() as i32);
            if !utf16_text(&class, n).starts_with("Chrome_WidgetWin") {
                return 1;
            }
            let mut title = [0u16; 256];
            let n = GetWindowTextW(hwnd, title.as_mut_ptr(), title.len() as i32);
            let title = utf16_text(&title, n);
            if title == "NROB Studio" || (!search.address.is_empty() && title.starts_with(&search.address)) {
                search.found = hwnd;
                return 0;
            }
            1
        }
    }
    let mut search = Search {
        address: ui_url.trim_start_matches("http://").trim_start_matches("https://").to_string(),
        found: std::ptr::null_mut(),
    };
    // SAFETY: the callback only reads window text and class into local buffers
    // and writes `search`, which lives across this synchronous call.
    unsafe { EnumWindows(Some(visit), &mut search as *mut Search as LPARAM) };
    (!search.found.is_null()).then_some(search.found)
}

/// Bring the UI window forward (restoring it if minimised), or open one.
pub fn show_ui(ui_url: &str) {
    if let Some(hwnd) = find_ui_window(ui_url) {
        // SAFETY: `hwnd` came from EnumWindows just now; these calls tolerate a
        // window that closed in between (they fail and return 0).
        unsafe {
            if IsIconic(hwnd) != 0 {
                ShowWindow(hwnd, SW_RESTORE);
            }
            BringWindowToTop(hwnd);
            SetForegroundWindow(hwnd);
        }
        return;
    }
    let mut last = LAST_LAUNCH.lock().unwrap_or_else(|p| p.into_inner());
    if last.is_some_and(|t| t.elapsed() < Duration::from_secs(4)) {
        return;
    }
    *last = Some(Instant::now());
    nrob_studio::open_ui(ui_url, "app");
}

const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
const RUN_VALUE: &str = "NROB Studio";

fn reg(args: &[&str]) -> bool {
    use std::os::windows::process::CommandExt;
    std::process::Command::new("reg")
        .args(args)
        .creation_flags(0x0800_0000)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Whether Windows starts the studio (in the tray, no window) at sign-in.
fn autostart() -> bool {
    reg(&["query", RUN_KEY, "/v", RUN_VALUE])
}

fn set_autostart(on: bool) {
    if on {
        let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default();
        let config = TRAY.get().map(|t| t.running.studio.config_path.display().to_string()).unwrap_or_default();
        let command = format!("\"{exe}\" --config \"{config}\" --open none {}", crate::AUTOSTART_FLAG);
        reg(&["add", RUN_KEY, "/v", RUN_VALUE, "/t", "REG_SZ", "/d", &command, "/f"]);
    } else {
        reg(&["delete", RUN_KEY, "/v", RUN_VALUE, "/f"]);
    }
}

fn menu(hwnd: HWND, tray: &Tray) {
    let (state, running) = llm_state(tray);
    let models = tray.running.studio.llm.models().join(", ");
    let status = if models.is_empty() { format!("Language model: {state}") } else { format!("Language model: {state} ({models})") };
    let items: Vec<(usize, String, u32)> = vec![
        (OPEN, "Open NROB Studio".into(), MF_STRING),
        (BROWSER, "Open in the browser".into(), MF_STRING),
        (0, String::new(), MF_SEPARATOR),
        (0, status, MF_STRING | MF_GRAYED),
        (TOGGLE_LLM, if running { "Stop the language model" } else { "Start the language model" }.into(), MF_STRING),
        (OUTPUTS, "Open the output folder".into(), MF_STRING),
        (0, String::new(), MF_SEPARATOR),
        (AUTOSTART, "Start with Windows".into(), MF_STRING | if autostart() { MF_CHECKED } else { MF_UNCHECKED }),
        (0, String::new(), MF_SEPARATOR),
        (QUIT, "Quit".into(), MF_STRING),
    ];
    let texts: Vec<Vec<u16>> = items.iter().map(|(_, t, _)| wide(t)).collect();
    let mut point = POINT { x: 0, y: 0 };
    // SAFETY: the menu handle is created and destroyed here; every item string is
    // a NUL-terminated buffer in `texts`, alive until DestroyMenu; `point` is a
    // valid out-parameter; `hwnd` is this thread's live window. The foreground
    // call and trailing WM_NULL are what Windows requires for tray menus to close.
    let chosen = unsafe {
        let hmenu = CreatePopupMenu();
        for ((id, _, flags), text) in items.iter().zip(&texts) {
            let text = if *flags & MF_SEPARATOR != 0 { std::ptr::null() } else { text.as_ptr() };
            AppendMenuW(hmenu, *flags, *id, text);
        }
        SetMenuDefaultItem(hmenu, OPEN as u32, 0);
        GetCursorPos(&mut point);
        SetForegroundWindow(hwnd);
        let chosen = TrackPopupMenu(hmenu, TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, point.x, point.y, 0, hwnd, std::ptr::null());
        PostMessageW(hwnd, WM_NULL, 0, 0);
        DestroyMenu(hmenu);
        chosen as usize
    };
    let studio = &tray.running.studio;
    match chosen {
        OPEN => open_ui(tray, "app"),
        BROWSER => open_ui(tray, "browser"),
        TOGGLE_LLM if running => studio.llm.stop(),
        TOGGLE_LLM => {
            let llm = Arc::clone(&studio.llm);
            let (cfg, root) = (studio.config(), studio.root.clone());
            std::thread::spawn(move || {
                if let Err(e) = llm.start(&cfg, &root) {
                    message_box(&format!("The language model did not start:\n{e}"));
                }
            });
        }
        OUTPUTS => {
            let dir = studio.output_root();
            let _ = std::fs::create_dir_all(&dir);
            nrob_studio::open_path(&dir.to_string_lossy());
        }
        AUTOSTART => set_autostart(!autostart()),
        QUIT => quit(hwnd),
        _ => {}
    }
}

fn quit(hwnd: HWND) {
    notify(hwnd, NIM_DELETE, None);
    if let Some(tray) = TRAY.get() {
        tray.running.studio.shutdown();
    }
    // SAFETY: called on the thread that owns the message loop.
    unsafe { PostQuitMessage(0) };
}

/// Every few seconds: refresh the tooltip and announce media that finished.
fn tick(hwnd: HWND) {
    let Some(tray) = TRAY.get() else { return };
    let balloon: Option<(String, String, bool)> = {
        let jobs = tray.running.studio.media.list();
        let mut seen = tray.announced.lock().unwrap_or_else(|p| p.into_inner());
        // Jobs the studio has let go of need no memory here either.
        seen.retain(|id| jobs.iter().any(|j| &j.id == id));
        let (mut ready, mut failed) = (Vec::new(), Vec::new());
        for job in jobs {
            if !job.finished() || seen.contains(&job.id) {
                continue;
            }
            seen.push(job.id.clone());
            match job.status.as_str() {
                "completed" => ready.push(job),
                "failed" => failed.push(job),
                _ => {}
            }
        }
        let what = |kind: &str| if kind == "video" { "Video" } else { "Image" };
        // Several finishing in one tick get one balloon that counts them.
        match (ready.as_slice(), failed.as_slice()) {
            ([], []) => None,
            ([j], []) => Some((format!("{} ready", what(j.kind.name())), j.prompt.chars().take(80).collect(), false)),
            ([], [j]) => Some((format!("{} failed", what(j.kind.name())), j.error.clone().unwrap_or_default().chars().take(200).collect(), true)),
            (r, []) => Some((format!("{} results ready", r.len()), "Open NROB Studio to see them in the Gallery.".into(), false)),
            (r, f) => Some((format!("{} ready, {} failed", r.len(), f.len()), "Open NROB Studio to see them in the Gallery.".into(), true)),
        }
    };
    match &balloon {
        Some((title, text, warn)) => notify(hwnd, NIM_MODIFY, Some((title, text, *warn))),
        None => notify(hwnd, NIM_MODIFY, None),
    }
}

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let taskbar = TASKBAR_CREATED.load(Ordering::Relaxed) as u32;
    match msg {
        CALLBACK => {
            if let Some(tray) = TRAY.get() {
                match lparam as u32 {
                    // One event per click: a double-click also sends two
                    // button-ups, and each used to open a window.
                    WM_LBUTTONUP => open_ui(tray, "app"),
                    WM_RBUTTONUP | WM_CONTEXTMENU => menu(hwnd, tray),
                    _ => {}
                }
            }
            0
        }
        WM_TIMER => {
            tick(hwnd);
            0
        }
        // Signing out or shutting down: stop the engines cleanly.
        WM_ENDSESSION if wparam != 0 => {
            if let Some(tray) = TRAY.get() {
                tray.running.studio.shutdown();
            }
            0
        }
        WM_DESTROY => {
            quit(hwnd);
            0
        }
        m if m != 0 && m == taskbar => {
            notify(hwnd, NIM_ADD, None);
            0
        }
        // SAFETY: forwarding the unmodified arguments Windows gave this procedure.
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

/// Show the icon and run the message loop until Quit.
pub fn run(running: Running, show_balloon: bool) {
    let refresh_autostart = autostart();
    let already: Vec<String> = running.studio.media.list().iter().filter(|j| j.finished()).map(|j| j.id.clone()).collect();
    let ui_url = running.ui_url.clone();
    let tray = Tray { running, announced: Mutex::new(already), icon: AtomicIsize::new(load_icon() as isize) };
    if TRAY.set(tray).is_err() {
        return;
    }
    let class = wide("NrobStudioTray");
    let title = wide("NROB Studio");
    let taskbar = wide("TaskbarCreated");
    // SAFETY: the class and title strings are NUL-terminated and outlive both
    // calls; `window_proc` has the WNDPROC signature; the window is never shown
    // (it exists to receive the icon's messages), and is created on this thread,
    // which then runs its message loop.
    let hwnd = unsafe {
        TASKBAR_CREATED.store(RegisterWindowMessageW(taskbar.as_ptr()) as isize, Ordering::Relaxed);
        let instance = GetModuleHandleW(std::ptr::null());
        let wc = WNDCLASSW {
            lpfnWndProc: Some(window_proc),
            hInstance: instance,
            lpszClassName: class.as_ptr(),
            ..Default::default()
        };
        RegisterClassW(&wc);
        CreateWindowExW(0, class.as_ptr(), title.as_ptr(), WS_OVERLAPPED, 0, 0, 0, 0, std::ptr::null_mut(), std::ptr::null_mut(), instance, std::ptr::null())
    };
    if hwnd.is_null() {
        message_box("Could not create the notification-area window.");
        return;
    }
    // Keep a registered "Start with Windows" pointing at this executable, with
    // today's command line.
    if refresh_autostart {
        set_autostart(true);
    }
    let hello = format!("Running in the background at {ui_url}. Click the icon to open it; right-click for more.");
    notify(hwnd, NIM_ADD, if show_balloon { Some(("NROB Studio", hello.as_str(), false)) } else { None });
    // SAFETY: `hwnd` is live and owned by this thread; no timer callback (the
    // WM_TIMER message is handled in window_proc).
    unsafe { SetTimer(hwnd, TIMER, 3000, None) };
    let mut msg: MSG = MSG { hwnd: std::ptr::null_mut(), message: 0, wParam: 0, lParam: 0, time: 0, pt: POINT { x: 0, y: 0 } };
    // SAFETY: `msg` is a valid MSG out-parameter for the loop's lifetime.
    unsafe {
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icon_sizes_come_from_the_file_directory() {
        let width = |size| {
            let (offset, _) = ico_image(ICON, size).unwrap();
            // Each image is a PNG (width at byte 16) or a DIB (width at byte 4).
            let img = &ICON[offset..];
            if img.starts_with(b"\x89PNG") {
                u32::from_be_bytes([img[16], img[17], img[18], img[19]]) as i32
            } else {
                i32::from_le_bytes([img[4], img[5], img[6], img[7]])
            }
        };
        assert_eq!(width(16), 16);
        assert!(width(20) >= 20, "a larger image for 125% scaling");
        assert!(width(32) >= 32);
        assert!(width(1000) >= 48, "the largest when none is big enough");
        assert!(ico_image(b"\0\0\x01\0\x01\0", 16).is_none());
    }
}
