/**
 * The "Install app" item of the project menu (see pwa/install.ts): shown only
 * while the browser offers to install the app, or on iOS Safari, where it says
 * how to do it by hand. Otherwise it is not there at all.
 */
import type { InstallController } from '../pwa/install';
import { h } from './dom';
import { modal } from './modal';

/** The one thing to tell someone on iOS Safari. */
export const IOS_INSTALL_HINT = 'Tap the Share button, then Add to Home Screen.';

export function installButton(install: InstallController): HTMLButtonElement {
  const button = h('button.install-app', { hidden: true }, 'Install app') as HTMLButtonElement;
  const show = (): void => {
    const state = install.state;
    button.hidden = state !== 'available' && state !== 'ios';
    button.title = state === 'ios' ? 'How to install OAIY on this device' : 'Install OAIY as an app: a window of its own, and it works offline';
  };
  button.addEventListener('click', () => {
    if (install.state === 'ios') void modal({ title: 'Install OAIY', message: IOS_INSTALL_HINT, ok: { label: 'OK', value: () => true }, cancel: 'Close' });
    else void install.install();
  });
  install.onChange(show);
  show();
  return button;
}
