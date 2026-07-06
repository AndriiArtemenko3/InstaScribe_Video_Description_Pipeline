import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import {
  Card,
  CardContent,
  CardHeader,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useAppStore } from '@/store/appStore'
import { Logo } from '@/components/ui/Logo'

interface LocationState {
  message?: string
}

export default function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const login = useAppStore((s) => s.login)

  const successMessage = (location.state as LocationState | null)?.message ?? null

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loginFailed, setLoginFailed] = useState(false)
  const [messageDismissed, setMessageDismissed] = useState(false)

  const isReady = email.length > 0 && password.length > 0
  const showSuccess = !!successMessage && !messageDismissed

  function handleEmailChange(value: string) {
    setEmail(value)
    setMessageDismissed(true)
    setLoginFailed(false)
  }

  function handlePasswordChange(value: string) {
    setPassword(value)
    setLoginFailed(false)
  }

  function handleSubmit() {
    const ok = login(email, password)
    if (ok) {
      navigate('/dashboard')
    } else {
      setLoginFailed(true)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-50">
      <Card className="w-form-field">
        <CardHeader className="space-y-2">
          <div className="inline-flex items-center gap-2">
          <Logo size={24} className="text-brand-400" />
            <span className="font-mono text-sm font-medium">InstaScribe</span>
          </div>
          <div className="space-y-1">
            <h1 className="text-lg font-medium">Welcome back</h1>
            <p className="text-sm text-muted-foreground">
              Sign in to your account to continue
            </p>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {showSuccess && (
            <p className="text-sm text-success-400">{successMessage}</p>
          )}

          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="email@example.com"
              value={email}
              onChange={(e) => handleEmailChange(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="password">Password</Label>
            </div>
            <Input
              id="password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => handlePasswordChange(e.target.value)}
            />
            <div className="text-right">
              <Link
                to="/forgot-password"
                className="text-sm text-muted-foreground hover:underline underline-offset-4"
              >
                Forgot password?
              </Link>
            </div>
          </div>

          <Button
            variant="default"
            className="w-full"
            disabled={!isReady}
            onClick={handleSubmit}
          >
            Sign in
          </Button>

          {loginFailed && (
            <p className="text-sm text-danger-400">
              Incorrect email or password.
            </p>
          )}

          <div className="flex flex-col items-center gap-4 pt-2">
            <hr className="border-neutral-200 w-full" />
            <p className="text-sm text-muted-foreground">
              Don't have an account?{' '}
              <Link
                to="/register"
                className="text-sm text-muted-foreground underline-offset-4 hover:underline"
              >
                Sign up
              </Link>
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
