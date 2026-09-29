import { memo, useMemo } from 'react';
import { MousePointerClick } from 'lucide-react';
import type { Node } from '@xyflow/react';
import { PropertyField } from 'oaiy-ui-components/nodes/fields/PropertyField';
import type { NodeDefinition } from 'oaiy-core';
import { BUNDLED_MODULES, getModuleLoader } from 'oaiy-core';
import { getCustomNodeDefinition } from '../../services/customNodeRegistry';
import { getService, listAllServices } from '../../utils/serviceRegistry';
import { listDesktopServices } from '../../lib/desktopServices';

import { SWATCH_CLASS } from './nodeSwatches';

export { SWATCH_CLASS };

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

    // If no node selected, say how to pick one.
    if (!selectedNode) {
        return (
            <div className={`p-3 ${className}`}>
                <div className="oaiy-empty bare">
                    <MousePointerClick size={22} />
                    <p className="oaiy-empty-title">No node selected</p>
                    <p className="oaiy-empty-text">Click a node on the canvas to set it up here.</p>
                </div>
            </div>
        );
    }

    if (!definition) {
        return (
            <div className={`p-3 ${className}`}>
                <div className="oaiy-note warn">
                    <strong>Unknown node type</strong>
                    <p className="mt-1 mb-0">
                        No definition for <code className="oaiy-code">{selectedNode.type}</code>. The flow was
                        likely saved with a node from a plugin that is no longer installed, or the node's id
                        was renamed in an update. Select it on the canvas and press Delete to remove it, or
                        replace it with a current one.
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
        <div className={`flex flex-col ${className}`}>
            {/* Which node: its kind's colour, its name, its id. */}
            <div className="flex items-center gap-2.5 border-b border-edge-secondary px-3.5 py-3">
                <span className={`h-8 w-1.5 shrink-0 rounded-full ${SWATCH_CLASS[definition.color || 'slate'] ?? SWATCH_CLASS.slate}`} />
                <div className="min-w-0">
                    <h3 className="m-0 truncate text-[13.5px] font-semibold text-content-primary">{definition.name}</h3>
                    <div className="truncate font-mono text-[11px] text-content-faint">{selectedNode.id}</div>
                </div>
            </div>

            {/* The node's settings. */}
            <div className="flex flex-col gap-4 p-3.5">
                {(!definition.properties || definition.properties.length === 0) && (
                    <p className="oaiy-help faint">This node has nothing to set.</p>
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
