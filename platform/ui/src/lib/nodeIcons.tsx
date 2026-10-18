/** The icons of the node kinds, by the names their definitions give them. */
import type React from 'react';
import {
  Bot, Box, Boxes, Brain, CircleArrowLeft, CircleArrowRight, CircleCheck, CirclePlay, CircleStop, Clock, Code, Database,
  Download, Eye, File as FileIcon, FileInput, FileText, Film, Folder, GitBranch, Globe, Grid2x2, Image as ImageIcon, Layers,
  Maximize2, Mic, MousePointer2, Music, Save, Scissors, Server, Trash2, Type, Video, Volume2, Workflow, Wrench, Zap,
  type LucideIcon,
} from 'lucide-react';

const NODE_ICONS: Record<string, LucideIcon> = {
  // Input
  type: Type, file: FileIcon, 'file-input': FileInput, 'file-text': FileText, folder: Folder,
  // AI
  bot: Bot, brain: Brain,
  // Flow control
  'git-branch': GitBranch, play: CirclePlay, stop: CircleStop, 'check-circle': CircleCheck, share: Workflow,
  'arrow-right-circle': CircleArrowRight, 'arrow-left-circle': CircleArrowLeft, cube: Boxes,
  // Utility, data, files
  tool: Wrench, code: Code, database: Database, save: Save, scissors: Scissors, trash: Trash2, server: Server,
  // Pictures, video, sound
  image: ImageIcon, eye: Eye, grid: Grid2x2, maximize: Maximize2, box: Box,
  video: Video, film: Film, layers: Layers, clock: Clock, download: Download,
  music: Music, 'volume-2': Volume2, mic: Mic,
  // Browser
  cursor: MousePointer2, zap: Zap, globe: Globe,
};

/**
 * A node kind's icon, by the name its definition gives it (`"icon": "brain"`): one set (lucide's, the editor's own
 * icons) for the palette, quick connect and the cards on the canvas (registered with oaiy-ui-components, see
 * bundled-web-modules.ts), so a node looks the same in each. Null for a name it does not know.
 */
export function nodeIconOf(iconName: string | undefined): React.ReactElement | null {
  const Icon = iconName ? NODE_ICONS[iconName] : undefined;
  return Icon ? <Icon className="w-4 h-4" size={16} strokeWidth={2} aria-hidden="true" /> : null;
}

/** A node kind's icon (`nodeIconOf`), or a box for a name it does not know. */
export function getNodeIcon(iconName: string | undefined): React.ReactElement {
  return nodeIconOf(iconName) ?? <Box className="w-4 h-4" size={16} strokeWidth={2} aria-hidden="true" />;
}
