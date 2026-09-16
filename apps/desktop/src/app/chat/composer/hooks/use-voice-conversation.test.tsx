import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BargeMonitorCallbacks } from '@/lib/voice-barge-in'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceConversation } from './use-voice-conversation'

// The full-duplex contract: the barge monitor is live across the WHOLE agent
// turn — generation (thinking) and playback (speaking) — so speaking over the
// model interrupts it mid-generation instead of the mic being deaf until TTS
// starts (the Windows report: interruption "never works" because the deaf
// window covered generation, and playback bleed made the old monitor's
// trigger unreachable).

const monitorCalls: BargeMonitorCallbacks[] = []
const stopMonitor = vi.fn()

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: (callbacks: BargeMonitorCallbacks) => {
    monitorCalls.push(callbacks)

    return stopMonitor
  }
}))

const markVoicePlaybackInterrupted = vi.fn()
const stopVoicePlayback = vi.fn()

vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: () => markVoicePlaybackInterrupted(),
  playSpeechText: vi.fn(async () => true),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: () => stopVoicePlayback()
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
}))

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording: false })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'mic failed',
          playbackFailed: 'playback failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

interface HookProps {
  busy: boolean
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(next => {
    resolve = next
  })

  return { promise, resolve }
}

function renderConversation(
  overrides: {
    beforeMicOpen?: () => Promise<void>
    onInterrupt?: () => void
    onSubmit?: (text: string) => Promise<void>
    onTranscribeAudio?: (audio: Blob) => Promise<string>
    transcript?: string
  } = {}
) {
  const onInterrupt = overrides.onInterrupt ?? vi.fn()

  // Mirrors the real app: submitting a turn makes the agent busy.
  const onBusyChange: { current: (busy: boolean) => void } = { current: () => undefined }

  const onSubmit = vi.fn(async (text: string) => {
    onBusyChange.current(true)
    await overrides.onSubmit?.(text)
  })

  const onStopWord = vi.fn()

  // First transcription is the turn that starts the conversation; subsequent
  // ones are barge captures (the overridable transcript).
  let transcriptions = 0

  const onTranscribeAudio = vi.fn(
    overrides.onTranscribeAudio ??
      (async () => (transcriptions++ === 0 ? 'kick off the task' : (overrides.transcript ?? 'and another thing')))
  )

  const hook = renderHook(
    ({ busy }: HookProps) =>
      useVoiceConversation({
        busy,
        beforeMicOpen: overrides.beforeMicOpen,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onInterrupt,
        onStopWord,
        onSubmit,
        onTranscribeAudio,
        pendingResponse: () => null
      }),
    { initialProps: { busy: false } }
  )

  onBusyChange.current = busy => hook.rerender({ busy })

  return { hook, onInterrupt, onStopWord, onSubmit, onTranscribeAudio }
}

/** Drive the hook into the generation phase (turn submitted, model working). */
async function enterThinking(hook: ReturnType<typeof renderConversation>['hook']) {
  await act(async () => {
    await hook.result.current.start()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('listening'))

  micHandle.stop.mockResolvedValueOnce({
    audio: new Blob(['q'], { type: 'audio/webm' }),
    durationMs: 900,
    heardSpeech: true
  })

  await act(async () => {
    hook.result.current.stopTurn()
  })
  await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
}

describe('useVoiceConversation full-duplex barge-in', () => {
  beforeEach(() => {
    monitorCalls.length = 0
    vi.clearAllMocks()
    micHandle.start.mockResolvedValue(undefined)
    micHandle.stop.mockResolvedValue(null)
  })

  afterEach(cleanup)

  it('arms the barge monitor during generation (before any reply audio exists)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)

    await waitFor(() => expect(hook.result.current.status).toBe('thinking'))
    // busy=true + thinking → the full-duplex monitor must be live.
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
  })

  it('interrupts the in-flight turn when speech trips mid-generation', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).toHaveBeenCalledTimes(1)
    expect(markVoicePlaybackInterrupted).toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('submits the captured interruption once the interrupt settles (busy clears)', async () => {
    const { hook, onSubmit } = renderConversation({ transcript: 'no, do it differently' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })

    // Interrupt lands → the turn ends → busy flips false.
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['x'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('no, do it differently'))
  })

  it('does not interrupt when speech trips during playback (turn already done)', async () => {
    const { hook, onInterrupt } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    // Turn finished; playback phase.
    hook.rerender({ busy: false })

    act(() => {
      monitorCalls.at(-1)?.onSpeech()
    })

    expect(onInterrupt).not.toHaveBeenCalled()
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('a spoken stop command in the barge capture ends the conversation instead of submitting', async () => {
    const { hook, onStopWord, onSubmit } = renderConversation({ transcript: 'stop' })

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const monitor = monitorCalls.at(-1)

    act(() => {
      monitor?.onSpeech()
    })
    hook.rerender({ busy: false })

    await act(async () => {
      monitor?.onUtterance?.(new Blob(['s'], { type: 'audio/webm' }))
    })

    await waitFor(() => expect(onStopWord).toHaveBeenCalledTimes(1))
    // Only the kickoff turn was submitted — the "stop" capture never was.
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalledWith('stop')
  })

  it('re-arms a single monitor per turn (idempotent ensure)', async () => {
    const { hook } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))

    const armed = monitorCalls.length

    // Effect re-runs (busy toggles, status changes) must not open more mics.
    hook.rerender({ busy: true })
    hook.rerender({ busy: true })

    expect(monitorCalls.length).toBe(armed)
  })

  it('drops a normal transcription that resolves after the conversation ends', async () => {
    const transcription = deferred<string>()

    const { hook, onSubmit, onTranscribeAudio } = renderConversation({
      onTranscribeAudio: () => transcription.promise
    })

    await act(async () => {
      await hook.result.current.start()
    })
    micHandle.stop.mockResolvedValueOnce({
      audio: new Blob(['voice'], { type: 'audio/webm' }),
      durationMs: 900,
      heardSpeech: true
    })

    act(() => hook.result.current.stopTurn())
    await waitFor(() => expect(onTranscribeAudio).toHaveBeenCalledTimes(1))

    await act(async () => {
      await hook.result.current.end()
      transcription.resolve('phantom transcript')
      await transcription.promise
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(hook.result.current.status).toBe('idle')
  })

  it('drops a barge transcription that resolves after the conversation ends', async () => {
    const bargeTranscription = deferred<string>()
    let transcriptionCount = 0

    const { hook, onSubmit, onTranscribeAudio } = renderConversation({
      onTranscribeAudio: async () => {
        if (transcriptionCount++ === 0) {
          return 'kick off the task'
        }

        return bargeTranscription.promise
      }
    })

    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
    const monitor = monitorCalls.at(-1)

    act(() => monitor?.onSpeech())
    hook.rerender({ busy: false })
    act(() => monitor?.onUtterance?.(new Blob(['barge'], { type: 'audio/webm' })))
    await waitFor(() => expect(onTranscribeAudio).toHaveBeenCalledTimes(2))

    await act(async () => {
      await hook.result.current.end()
      bargeTranscription.resolve('phantom interruption')
      await bargeTranscription.promise
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).not.toHaveBeenCalledWith('phantom interruption')
    expect(hook.result.current.status).toBe('idle')
  })

  it('does not open the microphone when start finishes after end', async () => {
    const micBarrier = deferred<void>()
    const { hook } = renderConversation({ beforeMicOpen: () => micBarrier.promise })

    let startPromise!: Promise<void>
    act(() => {
      startPromise = hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('idle'))

    await act(async () => {
      await hook.result.current.end()
      micBarrier.resolve()
      await startPromise
    })

    expect(micHandle.start).not.toHaveBeenCalled()
    expect(hook.result.current.status).toBe('idle')
  })

  it('serializes a restarted conversation behind the cancelled microphone start', async () => {
    const firstMicStart = deferred<undefined>()

    micHandle.start.mockImplementationOnce(() => firstMicStart.promise).mockResolvedValueOnce(undefined)
    const { hook } = renderConversation()

    let firstStart!: Promise<void>
    act(() => {
      firstStart = hook.result.current.start()
    })
    await waitFor(() => expect(micHandle.start).toHaveBeenCalledTimes(1))
    await act(async () => hook.result.current.end())

    let secondStart!: Promise<void>
    act(() => {
      secondStart = hook.result.current.start()
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    await act(async () => {
      firstMicStart.resolve(undefined)
      await firstStart
      await secondStart
    })

    expect(micHandle.start).toHaveBeenCalledTimes(2)
    expect(micHandle.cancel).toHaveBeenCalledTimes(2)
    const staleCancelOrder = micHandle.cancel.mock.invocationCallOrder.at(-1) ?? 0
    const restartedMicOrder = micHandle.start.mock.invocationCallOrder[1] ?? 0

    expect(staleCancelOrder).toBeLessThan(restartedMicOrder)
    expect(hook.result.current.status).toBe('listening')
  })

  it('does not restore thinking after a normal submit settles following end', async () => {
    const submission = deferred<void>()
    const { hook, onSubmit } = renderConversation({ onSubmit: () => submission.promise })

    await act(async () => hook.result.current.start())
    micHandle.stop.mockResolvedValueOnce({
      audio: new Blob(['voice'], { type: 'audio/webm' }),
      durationMs: 900,
      heardSpeech: true
    })
    act(() => hook.result.current.stopTurn())
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))

    await act(async () => {
      await hook.result.current.end()
      submission.resolve()
      await submission.promise
    })

    expect(hook.result.current.status).toBe('idle')
  })

  it('does not restore thinking after a barge submit settles following end', async () => {
    const submission = deferred<void>()

    const { hook, onSubmit } = renderConversation({
      onSubmit: text => (text === 'and another thing' ? submission.promise : Promise.resolve())
    })

    await enterThinking(hook)
    await waitFor(() => expect(monitorCalls.length).toBeGreaterThan(0))
    const monitor = monitorCalls.at(-1)

    act(() => monitor?.onSpeech())
    hook.rerender({ busy: false })
    act(() => monitor?.onUtterance?.(new Blob(['barge'], { type: 'audio/webm' })))
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(2))

    await act(async () => {
      await hook.result.current.end()
      submission.resolve()
      await submission.promise
    })

    expect(hook.result.current.status).toBe('idle')
  })
})
