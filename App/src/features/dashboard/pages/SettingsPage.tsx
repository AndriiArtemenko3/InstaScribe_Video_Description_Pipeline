import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAppStore } from '@/store/appStore'
import { Button } from '@/components/ui/button'
import {
  Database, ExternalLink, Mail, RefreshCw, User as UserIcon, Wifi,
  WifiOff, Coins, LogOut, BookOpen,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { ProviderPicker } from '../components/ProviderPicker'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8765'
const APP_VERSION = '0.1.0-dev'

type Connection = 'pending' | 'connected' | 'offline'

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-medium uppercase tracking-widest text-neutral-400">
      {children}
    </h2>
  )
}

function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <article className={cn('rounded-lg border border-neutral-200 bg-neutral-0 p-5', className)}>
      {children}
    </article>
  )
}

function FieldRow({
  label, value, hint, icon,
}: { label: string; value: React.ReactNode; hint?: string; icon?: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-3 first:pt-0 last:pb-0 border-b border-neutral-100 last:border-b-0">
      <div className="flex min-w-0 items-center gap-2">
        {icon && <span className="text-neutral-400">{icon}</span>}
        <div className="min-w-0">
          <p className="text-sm font-medium text-neutral-900">{label}</p>
          {hint && <p className="text-xs text-neutral-400">{hint}</p>}
        </div>
      </div>
      <div className="shrink-0 text-right text-sm text-neutral-600">{value}</div>
    </div>
  )
}

function countLocalEdits(): { keys: number; bytes: number } {
  let keys = 0
  let bytes = 0
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i)
    if (!k || !k.startsWith('instascribe:')) continue
    keys += 1
    bytes += (localStorage.getItem(k) ?? '').length
  }
  return { keys, bytes }
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export default function SettingsPage() {
  const navigate = useNavigate()
  const user = useAppStore((s) => s.currentUser)
  const logout = useAppStore((s) => s.logout)
  const projects = useAppStore((s) => s.projects)

  const [connection, setConnection] = useState<Connection>('pending')
  const [lastPingMs, setLastPingMs] = useState<number | null>(null)
  const [pinging, setPinging] = useState(false)
  const [localStats, setLocalStats] = useState(() => countLocalEdits())
  const [confirmClear, setConfirmClear] = useState(false)

  async function ping() {
    setPinging(true)
    const t0 = performance.now()
    try {
      const res = await fetch(`${API_BASE}/api/jobs`, { method: 'GET' })
      const dt = Math.round(performance.now() - t0)
      setConnection(res.ok ? 'connected' : 'offline')
      setLastPingMs(dt)
    } catch {
      setConnection('offline')
      setLastPingMs(null)
    } finally {
      setPinging(false)
    }
  }

  useEffect(() => { ping() }, [])

  function clearLocalEdits() {
    const toRemove: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k && k.startsWith('instascribe:') && k.endsWith(':edits')) toRemove.push(k)
    }
    toRemove.forEach((k) => localStorage.removeItem(k))
    setLocalStats(countLocalEdits())
    setConfirmClear(false)
  }

  function handleLogout() {
    logout()
    navigate('/login')
  }

  return (
    <main className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 max-w-3xl">
        <div className="flex flex-col gap-10">

          <section className="flex flex-col gap-4">
            <SectionHeading>Account</SectionHeading>
            <Card>
              {user ? (
                <>
                  <FieldRow
                    icon={<UserIcon size={14} />}
                    label="Name"
                    value={user.name}
                  />
                  <FieldRow
                    icon={<Mail size={14} />}
                    label="Email"
                    value={user.email}
                  />
                  <FieldRow
                    icon={<Coins size={14} />}
                    label="Token balance"
                    hint="Updates after each processed job."
                    value={user.tokenBalance != null
                      ? user.tokenBalance.toLocaleString()
                      : '—'}
                  />
                  <div className="pt-4">
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-2 text-danger-400 hover:text-danger-400"
                      onClick={handleLogout}
                    >
                      <LogOut size={14} /> Log out
                    </Button>
                  </div>
                </>
              ) : (
                <p className="text-sm text-neutral-500">Not signed in.</p>
              )}
            </Card>
          </section>

          <section className="flex flex-col gap-4">
            <SectionHeading>Backend</SectionHeading>
            <Card>
              <FieldRow
                icon={connection === 'connected'
                  ? <Wifi size={14} className="text-success-400" />
                  : connection === 'offline'
                    ? <WifiOff size={14} className="text-danger-400" />
                    : <Wifi size={14} />}
                label="Pipeline server"
                hint={connection === 'connected'
                  ? `Reachable in ${lastPingMs ?? '—'} ms`
                  : connection === 'offline'
                    ? 'Cannot reach the server — start it with `python3 server.py` in modular_pipeline/.'
                    : 'Checking…'}
                value={
                  <button
                    onClick={ping}
                    disabled={pinging}
                    className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium text-brand-500 hover:bg-brand-50 disabled:opacity-40 transition-colors"
                  >
                    <RefreshCw size={12} className={pinging ? 'animate-spin' : ''} />
                    Test
                  </button>
                }
              />
              <FieldRow
                label="Base URL"
                value={<code className="font-mono text-xs">{API_BASE}</code>}
              />
              <FieldRow
                label="Override"
                hint="Set VITE_API_BASE in App/.env.local to point at a non-local server."
                value={import.meta.env.VITE_API_BASE
                  ? <span className="text-success-400">set</span>
                  : <span className="text-neutral-400">default</span>}
              />
            </Card>
          </section>

          <section className="flex flex-col gap-4">
            <SectionHeading>Model provider</SectionHeading>
            <Card>
              <ProviderPicker />
            </Card>
          </section>

          <section className="flex flex-col gap-4">
            <SectionHeading>Local storage</SectionHeading>
            <Card>
              <FieldRow
                icon={<Database size={14} />}
                label="Projects in store"
                hint="Cached on this device; the server is still the source of truth."
                value={projects.length}
              />
              <FieldRow
                label="Cached scene edits"
                hint={`${localStats.keys} ${localStats.keys === 1 ? 'project' : 'projects'} · ${formatBytes(localStats.bytes)}`}
                value={
                  confirmClear ? (
                    <span className="inline-flex gap-1">
                      <Button variant="outline" size="sm" onClick={() => setConfirmClear(false)}>
                        Cancel
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        className="bg-danger-400 hover:bg-danger-400/90"
                        onClick={clearLocalEdits}
                      >
                        Clear all
                      </Button>
                    </span>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setConfirmClear(true)}
                      disabled={localStats.keys === 0}
                    >
                      Clear cached edits
                    </Button>
                  )
                }
              />
              <p className="pt-3 text-xs text-neutral-400">
                Server-side overrides survive — only this device's draft edits get removed.
              </p>
            </Card>
          </section>

          <section className="flex flex-col gap-4">
            <SectionHeading>About</SectionHeading>
            <Card>
              <FieldRow label="Version" value={APP_VERSION} />
              <FieldRow label="Environment" value={import.meta.env.DEV ? 'development' : 'production'} />
              <FieldRow
                label="Documentation"
                value={
                  <Link to="/dashboard/help" className="inline-flex items-center gap-1 text-brand-500 hover:underline">
                    <BookOpen size={12} /> Open help
                  </Link>
                }
              />
              <FieldRow
                label="OpenAI dashboard"
                hint="Token usage and billing live on OpenAI's side."
                value={
                  <a
                    href="https://platform.openai.com/usage"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-brand-500 hover:underline"
                  >
                    <ExternalLink size={12} /> Open
                  </a>
                }
              />
            </Card>
          </section>

        </div>
      </div>
    </main>
  )
}
