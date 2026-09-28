// No console window behind the app in a release build on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    bot_computer_lib::run()
}
