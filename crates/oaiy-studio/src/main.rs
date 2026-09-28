//! `oaiy-studio`: the studio as a console program. See the library docs, and
//! `oaiy-studio-tray` for the notification-area app.

#![forbid(unsafe_code)]

fn main() {
    let result = oaiy_studio::parse_args_from(std::env::args().skip(1)).and_then(|args| {
        if let Some(url) = oaiy_studio::running_instance(&args) {
            println!("oaiy-studio is already running at {url}; opening it");
            oaiy_studio::open_ui(&url, args.open.as_deref().unwrap_or("app"));
            return Ok(());
        }
        let running = oaiy_studio::launch(&args, false)?;
        oaiy_studio::open_ui(&running.ui_url, &running.open);
        oaiy_studio::console(&running.studio, &running.ui_url);
        Ok(())
    });
    if let Err(e) = result {
        eprintln!("oaiy-studio: {e}");
        std::process::exit(1);
    }
}
