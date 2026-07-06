export const queryKeys = {
  projects: () => ['projects'] as const,
  project: (id: string) => ['projects', id] as const,
  scenes: (projectId: string) => ['projects', projectId, 'scenes'] as const,
  entities: (projectId: string) => ['projects', projectId, 'entities'] as const,
  adGaps: (projectId: string) => ['projects', projectId, 'adGaps'] as const,
  audioEvents: (projectId: string) => ['projects', projectId, 'audioEvents'] as const,
  overrides: (projectId: string) => ['projects', projectId, 'overrides'] as const,
}
