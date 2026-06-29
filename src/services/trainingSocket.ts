import type { TrainingMessage } from '../types/training'

export class TrainingSocket {
  private ws: WebSocket | null = null

  connect(url: string, onMessage: (msg: TrainingMessage) => void): void {
    this.close()
    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      // Connection established — training will start automatically
    }

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as TrainingMessage
        onMessage(msg)
      } catch {
        console.warn('Oneiros: failed to parse WS message', event.data)
      }
    }

    this.ws.onerror = () => {
      onMessage({ type: 'error', message: 'WebSocket connection error — is the backend running?' })
    }

    this.ws.onclose = () => {
      // Normal closure — handled via message types
    }
  }

  close(): void {
    if (this.ws) {
      this.ws.onmessage = null
      this.ws.onerror = null
      this.ws.onclose = null
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close()
      }
      this.ws = null
    }
  }
}
