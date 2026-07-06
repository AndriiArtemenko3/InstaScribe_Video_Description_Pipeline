import type { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAppStore } from '@/store/appStore'
import { isStudyMode, isDemoBuild } from '@/lib/session'

interface GuardProps {
  children: ReactNode
}

export function AuthGuard({ children }: GuardProps) {
  const isAuthenticated = useAppStore((s) => s.isAuthenticated)
  if (isStudyMode() || isDemoBuild()) return <>{children}</>   // study/demo build: no login wall
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return <>{children}</>
}

export function GuestGuard({ children }: GuardProps) {
  const isAuthenticated = useAppStore((s) => s.isAuthenticated)
  if (isAuthenticated) return <Navigate to="/dashboard" replace />
  return <>{children}</>
}
