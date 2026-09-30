import './styles.css';
import { createContext } from './context';
import { mountEmbed } from './embed';

mountEmbed(document.getElementById('app') as HTMLElement, createContext());
