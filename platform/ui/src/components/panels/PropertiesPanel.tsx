import { memo, useMemo } from 'react';
import type { Node } from '@xyflow/react';
import { PropertyField } from 'oaiy-ui-components/nodes/fields/PropertyField';
import type { NodeDefinition } from 'oaiy-core';
import { BUNDLED_MODULES, getModuleLoader } from 'oaiy-core';
import { getCustomNodeDefinition } from '../../services/customNodeRegistry';
import { getService, listAllServices } from '../../utils/serviceRegistry';
import { listDesktopServices } from '../../lib/desktopServices';

// Static Tailwind literals for the node accent swatch, keyed by the node
// definition's `color`. Tailwind only emits classes it sees as whole literal
// strings during its content scan, so an interpolated `bg-${color}-500` purges
// to an invisible zero-width swatch for any color not statically present
// elsewhere. Listing the full literals here guarantees every node color paints.
export const SWATCH_CLASS: Record<string, string> = {
    amber: 'bg-amber-500', blue: 'bg-blue-500', cyan: 'bg-cyan-500',
    emerald: 'bg-emerald-500', gray: 'bg-gray-500', green: 'bg-green-500',
    indigo: 'bg-indigo-500', orange: 'bg-orange-500', pink: 'bg-pink-500',
    purple: 'bg-purple-500', red: 'bg-red-500', slate: 'bg-slate-500',
    teal: 'bg-teal-500', violet: 'bg-violet-500',
};

interface PropertiesPanelProps {
    selectedNode: Node | null;
    updateNodeData: (id: string, data: Record<string, unknown>) => void;
    className?: string;
}

const PropertiesPanel = memo(({ selectedNode, updateNodeData, className = '' }: PropertiesPanelProps) => {
    // Get definition from node data, or fallback to looking it up in the registry
    // This handles legacy nodes or nodes that lost their definition reference
    // NOTE: Hooks must be called unconditionally before any early returns
    const definition = useMemo((): NodeDefinition | undefined => {
        if (!selectedNode) return undefined;

        if (selectedNode.data.__definition) {
            return selectedNode.data.__definition as NodeDefinition;
        }

        // Fallback lookup in bundled modules
        for (const module of BUNDLED_MODULES) {
            const found = module.nodes.find(n => n.id === selectedNode.type);
            if (found) return found;
        }

        // Fallback lookup in custom node registry (for TypeScript-based embedded nodes)
        if (selectedNode.type) {
            const customDef = getCustomNodeDefinition(selectedNode.type);
            if (customDef) return customDef;

            // Fallback lookup in ModuleLoader (for path-based package nodes)
            const moduleLoader = getModuleLoader();
            const loaderDef = moduleLoader.getNodeDefinition(selectedNode.type);
            if (loaderDef) return loaderDef;
        }

        return undefined;
    }, [selectedNode?.data.__definition, selectedNode?.type, selectedNode]);

    // For a service_call node with a service picked, build a fallback
    // map of `{endpoint, method, headers, bodyTemplate, …}` lifted from
    // the linked service. Properties for which the node's own data is
    // null/undefined fall through to these so the panel shows the
    // service's effective config rather than blanks — matches what
    // happens on the canvas at run time, and lets users see at a
    // glance what a node inherits from its service vs what they've
    // overridden locally.
    //
    // Without this, a node where `service` was set programmatically
    // (e.g. by the WelcomeWizard's starter flow, or by an imported
    // package) shows empty endpoint / method / body fields until the
    // user re-picks the same service from the dropdown — which is
    // exactly the bug we're fixing here.
    const serviceFallbacks = useMemo<Record<string, unknown>>(() => {
        if (selectedNode?.type !== 'service_call') return {};
        const svcId = selectedNode.data.service;
        if (typeof svcId !== 'string' || svcId.length === 0) return {};
        const svc = getService(svcId);
        if (!svc) return {};
        const fb: Record<string, unknown> = {};
        if (svc.endpoint) fb.endpoint = svc.endpoint;
        if (svc.method) fb.method = svc.method;
        // Treat the literal "{}" the same as missing — it's the empty
        // placeholder the service form uses, and falling through to it
        // overwrites real per-node header overrides with nothing.
        if (svc.headers && svc.headers !== '{}') fb.headers = svc.headers;
        if (svc.bodyTemplate) fb.bodyTemplate = svc.bodyTemplate;
        if (svc.responseType) fb.responseType = svc.responseType;
        if (svc.responsePath !== undefined) fb.responsePath = svc.responsePath;
        if (svc.apiKeyConstant) fb.apiKeyConstant = svc.apiKeyConstant;
        if (svc.model) fb.model = svc.model;
        return fb;
    }, [selectedNode]);

    // Calculate all current values for conditional visibility logic.
    // Resolution order: explicit node-data override → service fallback
    // → property-schema default. Conditional-visibility predicates
    // (e.g. "show responsePath only when responseType=json") need the
    // effective value, not the raw stored value, otherwise a service
    // that ships responseType=json wouldn't reveal its responsePath
    // field until the user manually typed "json" into the dropdown.
    const allValues = useMemo(() => {
        if (!selectedNode || !definition) return {};

        const values: Record<string, unknown> = {};
        const props = definition.properties || [];
        for (const prop of props) {
            const own = selectedNode.data[prop.id];
            values[prop.id] = own ?? serviceFallbacks[prop.id] ?? prop.default;
        }
        return values;
    }, [definition, selectedNode, serviceFallbacks]);

    // If no node selected, show placeholder
    if (!selectedNode) {
        return (
            <div className={`flex flex-col items-center justify-center h-full text-slate-500 p-4 ${className}`}>
                <svg className="w-12 h-12 mb-2 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                </svg>
                <p className="text-sm font-medium">Select a node to view properties</p>
                <p className="text-xs mt-1">Click on any node in the canvas to edit its configuration.</p>
            </div>
        );
    }

    if (!definition) {
        return (
            <div className={`p-4 ${className}`}>
                <div className="bg-amber-100 dark:bg-amber-900/20 border border-amber-400 dark:border-amber-700/50 rounded-lg p-4 text-amber-900 dark:text-amber-200">
                    <h3 className="font-bold mb-1">Unknown Node Type</h3>
                    <p className="text-sm opacity-80">
                        Could not find definition for node type:{' '}
                        <code className="bg-amber-200/60 dark:bg-black/30 px-1 rounded">{selectedNode.type}</code>
                    </p>
                    <p className="text-xs opacity-70 mt-2">
                        This usually means the flow was saved with a node from a plugin
                        that is no longer installed, or the node id was renamed in an
                        update. Select the node on the canvas and press Delete to
                        remove it, or replace it with a current equivalent.
                    </p>
                </div>
            </div>
        );
    }

    const handleFieldChange = (fieldId: string) => (value: unknown) => {
        // Direct update via updateNodeData
        // We do NOT use selectedNode.data.onChange here because that handler is often
        // bound specifically to the 'value' property in OAIYBuilder (created via createHandler('value')).
        // Using it for other fields would incorrectly overwrite the 'value' property.
        //
        // Service-pick auto-populate: when the user changes the Service Preset
        // on a service_call node, copy the picked service's endpoint / body /
        // response / headers / method / apiKeyConstant into the node's data
        // alongside the id. Result: the inline fields in the Properties Panel
        // immediately show the service's effective config, and the user can
        // edit any of them as per-node overrides. Clearing the dropdown
        // (picking '') leaves the inline values alone — they're still active
        // until the user explicitly clears them.
        if (
            fieldId === 'service'
            && selectedNode.type === 'service_call'
            && typeof value === 'string'
            && value.length > 0
        ) {
            const svc = getService(value);
            if (svc) {
                const patch: Record<string, unknown> = { service: value };
                if (svc.endpoint) patch.endpoint = svc.endpoint;
                if (svc.method) patch.method = svc.method;
                if (svc.headers && svc.headers !== '{}') patch.headers = svc.headers;
                if (svc.bodyTemplate) patch.bodyTemplate = svc.bodyTemplate;
                if (svc.responseType) patch.responseType = svc.responseType;
                if (svc.responsePath !== undefined) patch.responsePath = svc.responsePath;
                if (svc.apiKeyConstant) patch.apiKeyConstant = svc.apiKeyConstant;
                // Pull the preset's default model in so the Model field
                // pre-populates with something useful instead of blank.
                // User can override at any time — their override wins
                // via the firstNonEmpty(data.model, preset?.model) check
                // in the service-call compiler.
                if (svc.model) patch.model = svc.model;
                updateNodeData(selectedNode.id, patch);
                return;
            }
        }

        // Model-edit sync into legacy body templates. New presets use
        // `{{model}}` substitution and don't need this — the runtime
        // swaps the placeholder at call time. But many existing
        // services still ship a literal model string in the template
        // (e.g. `"model": "llama3"`). If the user changes the Model
        // field from "llama3" → "llama2", we look for the OLD literal
        // in the template and substitute it in place. Safe no-op when
        // the template uses the placeholder form (the old literal
        // isn't there to replace).
        if (
            fieldId === 'model'
            && selectedNode.type === 'service_call'
            && typeof value === 'string'
        ) {
            const oldModel = (selectedNode.data.model as string | undefined) ?? serviceFallbacks.model as string | undefined;
            const currentTemplate = (selectedNode.data.bodyTemplate as string | undefined) ?? serviceFallbacks.bodyTemplate as string | undefined;
            const patch: Record<string, unknown> = { model: value };
            if (
                oldModel
                && oldModel !== value
                && currentTemplate
                && currentTemplate.includes(`"${oldModel}"`)
            ) {
                // JSON-quoted swap: `"llama3"` → `"llama2"`. Wrapping
                // in quotes anchors the match to a JSON string value
                // and avoids accidentally rewriting an unrelated
                // identical substring elsewhere in the template.
                patch.bodyTemplate = currentTemplate.split(`"${oldModel}"`).join(`"${value}"`);
            }
            updateNodeData(selectedNode.id, patch);
            return;
        }

        updateNodeData(selectedNode.id, { [fieldId]: value });
    };

    return (
        <div className={`flex flex-col h-full bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm ${className}`}>
            {/* Header */}
            <div className="p-3 border-b border-slate-200/50 dark:border-slate-700/50 flex items-center gap-2 bg-slate-100/30 dark:bg-slate-800/30">
                <div className={`w-2 h-8 rounded-full ${SWATCH_CLASS[definition.color || 'slate'] ?? 'bg-slate-500'}`} />
                <div>
                    <h2 className="font-bold text-slate-800 dark:text-slate-100 text-sm">{definition.name}</h2>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">{selectedNode.id}</div>
                </div>
            </div>

            {/* Properties Scroll Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                {(!definition.properties || definition.properties.length === 0) && (
                    <p className="text-slate-400 dark:text-slate-500 text-sm italic">No properties to configure.</p>
                )}

                {(definition.properties || [])
                    // Hide the Service Preset dropdown when there are no
                    // services to pick at all — an empty picker is just noise.
                    // Count BOTH the user's saved services AND any the OAIY
                    // OAIY Desktop is currently exposing; otherwise a running
                    // companion service is unpickable for a user who hasn't
                    // saved one of their own.
                    .filter((prop) => {
                        if (prop.id !== 'service') return true;
                        return listAllServices().length > 0 || listDesktopServices().length > 0;
                    })
                    .map((prop) => {
                        // Same resolution order as `allValues` above: node
                        // override → service fallback → schema default.
                        // The fallback is read-only — the user typing
                        // anything into the field calls handleFieldChange
                        // and stores the new value in node data, which
                        // then wins over the fallback on the next render.
                        const own = selectedNode.data[prop.id];
                        const displayValue = own ?? serviceFallbacks[prop.id];
                        return (
                            <PropertyField
                                key={prop.id}
                                property={prop}
                                value={displayValue}
                                onChange={handleFieldChange(prop.id)}
                                allValues={allValues}
                            />
                        );
                    })}
            </div>
        </div>
    );
});

PropertiesPanel.displayName = 'PropertiesPanel';

export default PropertiesPanel;
