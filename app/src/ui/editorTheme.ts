/**
 * The editor's look, from the page's colours (styles.css's tokens, which
 * follow light and dark): its background, gutter, selection, active line and
 * cursor, and the code's colours, in JetBrains Mono.
 */
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import type { Extension } from '@codemirror/state';
import { EditorView } from '@codemirror/view';
import { tags as t } from '@lezer/highlight';

const spec = {
  '&': { color: 'var(--text)', backgroundColor: 'var(--editor-bg)', height: '100%' },
  '&.cm-focused': { outline: 'none' },
  '.cm-scroller': { fontFamily: 'var(--mono)', fontSize: '12.5px', lineHeight: '1.65' },
  '.cm-content': { caretColor: 'var(--accent)', padding: '10px 0' },
  '.cm-line': { padding: '0 14px 0 6px' },
  '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--accent)', borderLeftWidth: '2px' },
  '.cm-selectionBackground, .cm-content ::selection': { backgroundColor: 'var(--editor-selection) !important' },
  '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground': { backgroundColor: 'var(--editor-selection-focused)' },
  '.cm-activeLine': { backgroundColor: 'var(--editor-active-line)' },
  '.cm-gutters': { backgroundColor: 'var(--editor-bg)', color: 'var(--editor-gutter-text)', border: 'none' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--text)' },
  '.cm-lineNumbers .cm-gutterElement': { padding: '0 8px 0 14px', minWidth: '40px' },
  '.cm-foldGutter .cm-gutterElement': { color: 'var(--editor-gutter-text)', padding: '0 4px' },
  '.cm-foldPlaceholder': { backgroundColor: 'var(--panel-2)', border: '1px solid var(--border)', color: 'var(--muted)', borderRadius: '4px', padding: '0 5px' },
  '.cm-matchingBracket, &.cm-focused .cm-matchingBracket': { backgroundColor: 'var(--editor-bracket)', outline: '1px solid color-mix(in srgb, var(--accent) 40%, transparent)' },
  '.cm-nonmatchingBracket': { color: 'var(--err)' },
  '.cm-searchMatch': { backgroundColor: 'color-mix(in srgb, var(--warn) 28%, transparent)', outline: '1px solid color-mix(in srgb, var(--warn) 55%, transparent)' },
  '.cm-searchMatch.cm-searchMatch-selected': { backgroundColor: 'color-mix(in srgb, var(--warn) 45%, transparent)' },
  '.cm-selectionMatch': { backgroundColor: 'color-mix(in srgb, var(--accent) 12%, transparent)' },
  '.cm-tooltip': { backgroundColor: 'var(--panel)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: '8px', boxShadow: 'var(--shadow-soft)' },
  '.cm-tooltip-autocomplete > ul > li[aria-selected]': { backgroundColor: 'var(--selected)', color: 'var(--text)' },
  '.cm-panels': { backgroundColor: 'var(--panel)', color: 'var(--text)', fontFamily: 'var(--sans)' },
  '.cm-panels.cm-panels-top': { borderBottom: '1px solid var(--border)' },
  '.cm-panels.cm-panels-bottom': { borderTop: '1px solid var(--border)' },
  '.cm-textfield': { backgroundColor: 'var(--panel-2)', border: '1px solid var(--border)', borderRadius: '6px', color: 'var(--text)' },
  '.cm-button': { backgroundImage: 'none', backgroundColor: 'var(--panel-2)', border: '1px solid var(--border)', borderRadius: '6px', color: 'var(--text)' },
};

const code = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.moduleKeyword, t.operatorKeyword, t.definitionKeyword, t.modifier], color: 'var(--syn-keyword)' },
  { tag: [t.string, t.special(t.string), t.regexp, t.character], color: 'var(--syn-string)' },
  { tag: [t.number, t.bool, t.null, t.atom], color: 'var(--syn-number)' },
  { tag: [t.comment, t.lineComment, t.blockComment, t.docComment], color: 'var(--syn-comment)', fontStyle: 'italic' },
  { tag: [t.function(t.variableName), t.function(t.propertyName), t.macroName], color: 'var(--syn-function)' },
  { tag: [t.typeName, t.className, t.namespace, t.definition(t.typeName)], color: 'var(--syn-type)' },
  { tag: [t.propertyName, t.attributeName], color: 'var(--syn-property)' },
  { tag: [t.tagName, t.self], color: 'var(--syn-tag)' },
  { tag: [t.definition(t.variableName), t.labelName], color: 'var(--syn-definition)' },
  { tag: [t.operator, t.punctuation, t.separator, t.bracket], color: 'var(--syn-punctuation)' },
  { tag: [t.meta, t.processingInstruction, t.annotation], color: 'var(--syn-meta)' },
  { tag: [t.heading], color: 'var(--syn-heading)', fontWeight: '700' },
  { tag: [t.strong], fontWeight: '700' },
  { tag: [t.emphasis], fontStyle: 'italic' },
  { tag: [t.link, t.url], color: 'var(--syn-link)', textDecoration: 'underline' },
  { tag: [t.invalid], color: 'var(--err)' },
]);

const light: Extension = [EditorView.theme(spec, { dark: false }), syntaxHighlighting(code)];
const dark: Extension = [EditorView.theme(spec, { dark: true }), syntaxHighlighting(code)];

/** The editor's look for light or dark (the colours come from the page; this tells CodeMirror which it is). */
export function editorLook(theme: 'light' | 'dark'): Extension {
  return theme === 'dark' ? dark : light;
}
