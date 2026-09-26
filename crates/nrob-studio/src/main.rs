//! `nrob-studio`: the studio as a console program. See the library docs, and
//! `nrob-studio-tray` for the notification-area app.

#![forbid(unsafe_code)]

fn main() {
    let result = nrob_studio::parse_args_from(std::env::args().skip(1)).and_then(|args| {
        if let Some(url) = nrob_studio::running_instance(&args) {
            println!("nrob-studio is already running at {url}; opening it");
            nrob_studio::open_ui(&url, args.open.as_deref().unwrap_or("app"));
            return Ok(());
        }
        let running = nrob_studio::launch(&args, false)?;
        nrob_studio::open_ui(&running.ui_url, &running.open);
        nrob_studio::console(&running.studio, &running.ui_url);
        Ok(())
    });
    if let Err(e) = result {
        eprintln!("nrob-studio: {e}");
        std::process::exit(1);
    }
}
