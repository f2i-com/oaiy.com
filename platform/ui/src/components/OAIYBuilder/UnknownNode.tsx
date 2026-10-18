/**
 * A node of a kind this editor does not have (a plugin's that is not installed, a node only OAIY Desktop runs), as
 * React Flow draws one with no component of its type: its default. That default showed only the node's label, so a
 * node with none was a dashed box saying "Unknown node" and nothing of what it was; this says its type as well, which
 * is what Properties and the docs call it. (Its look is index.css's `.react-flow__node-default`.)
 */
import { memo } from 'react';
import { Handle, Position, useStore, type NodeProps } from '@xyflow/react';

function UnknownNode({ id, data }: NodeProps) {
  // (React Flow passes this component "default" as its type: the node's own is in the store)
  const type = useStore((st) => st.nodeLookup.get(id)?.type);
  const label = typeof (data as { label?: unknown })?.label === 'string' ? ((data as { label: string }).label).trim() : '';
  return (
    <>
      <Handle type="target" position={Position.Top} />
      <div className="oaiy-unknown-node" title={type ? `A ${type} node: this editor has none of that kind` : undefined}>
        {label && <strong>{label}</strong>}
        {type && <code>{type}</code>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </>
  );
}

export default memo(UnknownNode);
