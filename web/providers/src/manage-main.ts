import './styles.css';
import { createContext } from './context';
import { mountManage } from './ui';

mountManage(document.getElementById('app') as HTMLElement, createContext());

// Ask the browser not to evict this origin's storage (the keys are in it). Only the top-level Providers page asks, never the hidden
// frame or the modal: Firefox puts a prompt in front of the person, and a page they did not open should not raise one.
if (window.top === window) void navigator.storage?.persist?.().catch(() => false);
