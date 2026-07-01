import logoUrl from '../assets/logo.png'

interface AppLogoProps {
  size?: number
}

export function AppLogo({ size = 24 }: AppLogoProps) {
  return (
    <img
      src={logoUrl}
      alt="Oneiros"
      width={size}
      height={size}
      style={{ display: 'block', objectFit: 'contain', flexShrink: 0 }}
    />
  )
}
