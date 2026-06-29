import type { AppNode, AppEdge } from '../../types/graph'

/**
 * Topological sort via Kahn's algorithm.
 * Returns nodes in dependency order (sources first), or null if a cycle exists.
 */
export function topologicalSort(
  nodes: AppNode[],
  edges: AppEdge[]
): AppNode[] | null {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))
  const inDegree = new Map<string, number>()
  const adj = new Map<string, string[]>()

  for (const node of nodes) {
    inDegree.set(node.id, 0)
    adj.set(node.id, [])
  }

  for (const edge of edges) {
    adj.get(edge.source)?.push(edge.target)
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1)
  }

  const queue: string[] = []
  for (const [id, degree] of inDegree) {
    if (degree === 0) queue.push(id)
  }

  const sorted: AppNode[] = []

  while (queue.length > 0) {
    const id = queue.shift()!
    const node = nodeMap.get(id)
    if (node) sorted.push(node)

    for (const neighbor of adj.get(id) ?? []) {
      const deg = (inDegree.get(neighbor) ?? 0) - 1
      inDegree.set(neighbor, deg)
      if (deg === 0) queue.push(neighbor)
    }
  }

  return sorted.length === nodes.length ? sorted : null
}
