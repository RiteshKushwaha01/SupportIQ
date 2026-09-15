'use client'

import { useState } from 'react'

type Source = {
  customer_query: string
  agent_response: string
  similarity: number
}

type ChatResponse = {
  query: string
  answer: string | null
  intent: string
  intent_score: number
  decision: string
  escalate: boolean
  reason: string
  risk_level: string
  risk_categories: string[]
  confidence: number
  confidence_level: string
  max_similarity: number | null
  average_similarity: number | null
  retrieved_results: Source[]
}

export default function Home() {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const sendMessage = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError('')
    setResponse(null)

    try {
      const res = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          top_k: 5,
        }),
      })

      if (!res.ok) {
        throw new Error('Unable to connect to SupportIQ API')
      }

      const data: ChatResponse = await res.json()
      setResponse(data)
      setQuery('')
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Something went wrong while processing the request.',
      )
    } finally {
      setLoading(false)
    }
  }

  const confidencePercent = response ? response.confidence * 100 : 0
  const intentScorePercent = response ? response.intent_score * 100 : 0
  const isHuman = response?.escalate === true

  const formatIntent = (intent: string) => {
    return intent
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  return (
    <main className="font-sans flex h-full min-h-0 overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <aside className="hidden h-full min-h-0 w-[210px] shrink-0 flex-col border-r border-white/[0.06] bg-[#0a1019] lg:flex">
        <div className="border-b border-white/[0.06] px-5 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white text-sm font-black text-slate-950">
              S
            </div>
            <div>
              <h1 className="text-[15px] font-bold tracking-tight">
                SupportIQ
              </h1>
              <p className="text-[11px] text-slate-500">Support Intelligence</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 py-5">
          <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
            Workspace
          </p>
          <div className="space-y-0.5">
            <button className="flex w-full items-center gap-2.5 rounded-lg bg-white/[0.08] px-3 py-2.5 text-[13px] font-medium text-white">
              <span className="text-sm opacity-70">▣</span>
              Inbox
              <span className="ml-auto rounded bg-white px-1.5 py-0.5 text-[10px] font-bold text-slate-950">
                1
              </span>
            </button>
            <button className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-[13px] text-slate-400 transition hover:bg-white/[0.04] hover:text-white">
              <span className="opacity-60">◉</span>
              Conversations
            </button>
            <button className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-[13px] text-slate-400 transition hover:bg-white/[0.04] hover:text-white">
              <span className="opacity-60">◌</span>
              Analytics
            </button>
            <button className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-[13px] text-slate-400 transition hover:bg-white/[0.04] hover:text-white">
              <span className="opacity-60">◇</span>
              Knowledge Base
            </button>
          </div>

          <p className="mb-2 mt-8 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
            System
          </p>
          <div className="space-y-0.5">
            <button className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-[13px] text-slate-400 transition hover:bg-white/[0.04] hover:text-white">
              <span className="opacity-60">⚙</span>
              Settings
            </button>
            <button className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-[13px] text-slate-400 transition hover:bg-white/[0.04] hover:text-white">
              <span className="opacity-60">?</span>
              Help
            </button>
          </div>

          {/* Gemini usage notice */}
          <div className="mt-4 rounded-lg border border-amber-900/50 bg-amber-950/20 px-3 py-2.5">
            <div className="flex items-start gap-2">
              <span className="mt-0.5 shrink-0 text-[10px] text-amber-400">
                ⚠
              </span>
              <div className="min-w-0">
                <p className="text-[10px] font-semibold text-amber-300">
                  Demo & API usage notice
                </p>
                <p className="mt-1 text-[9px] leading-relaxed text-amber-200/60">
                  This project uses the Gemini API for response generation.
                  Free-tier requests are subject to quota and rate limits. Avoid
                  repeatedly sending the same question during testing.
                </p>
                <p className="mt-1.5 text-[9px] leading-relaxed text-slate-500">
                  For development without consuming Gemini quota, set{' '}
                  <code className="rounded bg-slate-900 px-1 py-0.5 text-[8px] text-slate-300">
                    MOCK_LLM=true
                  </code>{' '}
                  in your{' '}
                  <code className="text-[8px] text-slate-400">.env</code> file.
                </p>
              </div>
            </div>
          </div>
        </nav>

        <div className="border-t border-white/[0.06] p-3">
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
            <div className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              <span className="text-xs font-medium text-slate-300">
                System operational
              </span>
            </div>
            <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
              Retrieval and escalation services are running normally.
            </p>
          </div>
        </div>
      </aside>

      {/* Main workspace */}
      <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="flex shrink-0 items-center justify-between border-b border-white/[0.06] bg-[#0a1019]/80 px-5 py-3.5 backdrop-blur-sm lg:px-6">
          <div>
            <p className="text-[10px] font-medium uppercase tracking-widest text-slate-500">
              Agent Workspace
            </p>
            <h2 className="mt-0.5 text-base font-semibold">
              Customer Support Inbox
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden text-right sm:block">
              <p className="text-sm font-medium">AI Support Agent</p>
              <p className="text-[11px] text-slate-500">AmazonHelp</p>
            </div>
            <div className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.06] text-xs font-semibold">
              AI
            </div>
          </div>
        </header>

        {/* Stats row */}
        <div className="grid shrink-0 grid-cols-3 gap-2 border-b border-white/[0.06] px-4 py-3 sm:gap-3 sm:px-5 lg:px-6">
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">
              Processing Mode
            </p>
            <div className="mt-2 flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span className="text-sm font-semibold">RAG Active</span>
            </div>
          </div>
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">
              Knowledge Sources
            </p>
            <p className="mt-2 text-sm font-semibold">148,556</p>
            <p className="text-[10px] text-slate-500">Support conversations</p>
          </div>
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">
              Retrieval
            </p>
            <p className="mt-2 text-sm font-semibold">FAISS</p>
            <p className="text-[10px] text-slate-500">Top-5 semantic search</p>
          </div>
        </div>

        {/* Chat + analysis */}
        <div className="relative grid min-h-0 flex-1 gap-0 overflow-hidden lg:grid-cols-[60vw_minmax(0,1fr)]">
          {/* Conversation panel */}
          <section className="flex min-h-0 min-w-0 flex-col border-r border-white/[0.06] bg-[#0c121c]">
            {/* Chat header */}
            <div className="flex shrink-0 items-center justify-between border-b border-white/[0.06] px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-blue-500/30 to-blue-600/10 text-sm font-semibold ring-2 ring-blue-500/20">
                    C
                  </div>
                  <span className="absolute bottom-0 right-0 h-2.5 w-2.5 rounded-full border-2 border-[#0c121c] bg-emerald-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold">Customer Conversation</p>
                  <p className="text-xs text-slate-500">
                    Live AI-assisted support
                  </p>
                </div>
              </div>
              <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] text-slate-400">
                New conversation
              </span>
            </div>

            {/* Messages */}
            <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 py-6">
              {!response && !loading && (
                <div className="flex h-full min-h-[280px] flex-col items-center justify-center text-center">
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500/20 to-violet-500/10 text-xl ring-1 ring-white/10">
                    ✦
                  </div>
                  <h3 className="text-base font-semibold">Ready to assist</h3>
                  <p className="mt-2 max-w-sm text-[13px] leading-relaxed text-slate-500">
                    Enter a customer question below. SupportIQ will retrieve
                    relevant historical conversations, assess risk, and decide
                    whether AI or a human agent should handle the request.
                  </p>
                </div>
              )}

              {loading && (
                <div className="flex h-full min-h-[280px] flex-col items-center justify-center">
                  <div className="flex items-center gap-1.5 rounded-2xl border border-white/[0.06] bg-white/[0.03] px-5 py-4">
                    <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
                  </div>
                  <p className="mt-4 text-sm text-slate-400">
                    Analyzing customer request...
                  </p>
                  <p className="mt-1 text-xs text-slate-600">
                    Retrieving relevant support conversations
                  </p>
                </div>
              )}

              {response && (
                <div className="w-full space-y-5">
                  {/* Customer message */}
                  <div className="flex gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-700/80 text-[11px] font-semibold">
                      C
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="mb-1.5 flex items-baseline gap-2">
                        <span className="text-xs font-semibold text-slate-300">
                          Customer
                        </span>
                        <span className="text-[10px] text-slate-600">
                          Just now
                        </span>
                      </div>
                      <div className="inline-block max-w-full rounded-2xl rounded-tl-md bg-[#1a2744] px-4 py-3 text-[13px] leading-relaxed text-slate-100 ring-1 ring-blue-500/10">
                        {response.query}
                      </div>
                    </div>
                  </div>

                  {/* Decision banner — system message */}
                  <div
                    className={`mx-2 rounded-xl border px-4 py-3.5 ${
                      isHuman
                        ? 'border-red-500/20 bg-red-500/[0.06]'
                        : 'border-emerald-500/20 bg-emerald-500/[0.06]'
                    }`}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="flex items-start gap-3">
                        <div
                          className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-sm ${
                            isHuman
                              ? 'bg-red-500/15 text-red-400'
                              : 'bg-emerald-500/15 text-emerald-400'
                          }`}
                        >
                          {isHuman ? '!' : '✓'}
                        </div>
                        <div>
                          <p
                            className={`text-[13px] font-semibold ${
                              isHuman ? 'text-red-400' : 'text-emerald-400'
                            }`}
                          >
                            {isHuman
                              ? 'Human escalation required'
                              : 'AI can handle this request'}
                          </p>
                          <p className="mt-0.5 text-[11px] text-slate-500">
                            Decision engine: {response.reason}
                          </p>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        <span className="rounded-full bg-black/20 px-2.5 py-1 text-[11px] text-slate-300">
                          Confidence {confidencePercent.toFixed(1)}%
                        </span>
                        <span
                          className={`rounded-full px-2.5 py-1 text-[11px] ${
                            isHuman
                              ? 'bg-red-500/10 text-red-400'
                              : 'bg-emerald-500/10 text-emerald-400'
                          }`}
                        >
                          Risk: {response.risk_level}
                        </span>
                      </div>
                    </div>

                    {response.risk_categories.length > 0 && (
                      <div className="mt-3 border-t border-red-500/10 pt-3">
                        <p className="text-[11px] text-slate-500">
                          Detected risk categories
                        </p>
                        <div className="mt-1.5 flex flex-wrap gap-1.5">
                          {response.risk_categories.map((category) => (
                            <span
                              key={category}
                              className="rounded-md bg-red-500/10 px-2 py-0.5 text-[11px] text-red-400"
                            >
                              {category}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* AI response */}
                  <div className="flex gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white text-[11px] font-black text-slate-950">
                      S
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="mb-1.5 flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-300">
                          SupportIQ
                        </span>
                        <span className="rounded bg-white/[0.08] px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider text-slate-500">
                          AI
                        </span>
                      </div>
                      <div className="inline-block max-w-full rounded-2xl rounded-tl-md border border-white/[0.06] bg-white/[0.04] px-4 py-3 text-[13px] leading-relaxed text-slate-200">
                        {response.answer ||
                          'This conversation should be handled by a human support agent.'}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Composer */}
            <div className="shrink-0 border-t border-white/[0.06] bg-[#0a1019]/90 p-4 backdrop-blur-sm">
              <div className="flex w-full gap-2">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      sendMessage()
                    }
                  }}
                  placeholder="Type a customer message..."
                  className="min-w-0 flex-1 rounded-xl border border-white/[0.08] bg-[#080c14] px-4 py-3 text-[13px] text-white outline-none placeholder:text-slate-600 transition focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10"
                />
                <button
                  onClick={sendMessage}
                  disabled={loading || !query.trim()}
                  className="shrink-0 rounded-xl bg-white px-5 py-3 text-[13px] font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading ? '...' : 'Send'}
                </button>
              </div>
              <p className="mt-2 px-1 text-center text-[10px] text-slate-600">
                Press Enter to send • SupportIQ analyzes every request before
                generating a response.
              </p>
            </div>
          </section>

          {/* AI Analysis panel */}
          <aside className="hidden min-h-0 overflow-y-auto overscroll-contain bg-[#0a1019] p-4 lg:block lg:space-y-4">
            {/* Intent classification */}
            <section className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-slate-500">
                    Intent Classification
                  </p>
                  <h3 className="mt-0.5 text-sm font-semibold">
                    Customer Intent
                  </h3>
                </div>

                <span className="text-[10px] text-slate-600">BGE Semantic</span>
              </div>

              {response ? (
                <>
                  <div className="mt-5">
                    <p className="text-[9px] uppercase tracking-wider text-slate-600">
                      Detected Intent
                    </p>

                    <p className="mt-1.5 text-lg font-semibold text-slate-200">
                      {formatIntent(response.intent)}
                    </p>
                  </div>

                  <div className="mt-4">
                    <div className="flex items-center justify-between">
                      <p className="text-[9px] uppercase tracking-wider text-slate-600">
                        Semantic Score
                      </p>

                      <p className="text-sm font-semibold text-slate-300">
                        {intentScorePercent.toFixed(1)}%
                      </p>
                    </div>

                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-violet-500 to-blue-500 transition-all duration-500"
                        style={{
                          width: `${Math.min(intentScorePercent, 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                </>
              ) : (
                <p className="mt-5 text-[13px] text-slate-600">
                  Intent classification will appear after a request.
                </p>
              )}
            </section>

            {/* Routing decision */}
            <section className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-5">
              <p className="text-[10px] uppercase tracking-wider text-slate-500">
                Routing Decision
              </p>

              {response ? (
                <div className="mt-4">
                  <div
                    className={`flex items-center gap-3 rounded-lg p-3.5 ${
                      isHuman ? 'bg-red-500/[0.08]' : 'bg-emerald-500/[0.08]'
                    }`}
                  >
                    <div
                      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-base ${
                        isHuman
                          ? 'bg-red-500/15 text-red-400'
                          : 'bg-emerald-500/15 text-emerald-400'
                      }`}
                    >
                      {isHuman ? '!' : '✓'}
                    </div>
                    <div>
                      <p
                        className={`text-sm font-semibold ${
                          isHuman ? 'text-red-400' : 'text-emerald-400'
                        }`}
                      >
                        {isHuman ? 'Human Escalation' : 'AI Handled'}
                      </p>
                      <p className="mt-0.5 text-[11px] text-slate-500">
                        {response.reason}
                      </p>
                    </div>
                  </div>
                  <div className="mt-3 space-y-2">
                    <div className="flex justify-between text-[13px]">
                      <span className="text-slate-500">Risk level</span>
                      <span className="font-medium capitalize">
                        {response.risk_level}
                      </span>
                    </div>
                    <div className="flex justify-between text-[13px]">
                      <span className="text-slate-500">Decision</span>
                      <span className="font-medium capitalize">
                        {response.decision}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="mt-4 text-[13px] text-slate-600">
                  No routing decision yet.
                </p>
              )}
            </section>

            {/* Sources */}
            <section className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-slate-500">
                    Knowledge Retrieval
                  </p>
                  <h3 className="mt-0.5 text-sm font-semibold">
                    Top Support Sources
                  </h3>
                </div>
                {response && (
                  <span className="rounded-full bg-white/[0.06] px-2 py-0.5 text-[10px] text-slate-400">
                    {response.retrieved_results.length} results
                  </span>
                )}
              </div>

              {response ? (
                <div className="mt-4 space-y-2.5">
                  {response.retrieved_results.map((source, index) => (
                    <div
                      key={index}
                      className="rounded-lg border border-white/[0.05] bg-black/20 p-3.5 transition hover:border-white/10"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-[9px] font-semibold uppercase tracking-wider text-slate-600">
                          Source {index + 1}
                        </span>
                        <span className="text-[11px] font-semibold text-slate-300">
                          {(source.similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="mt-2 line-clamp-3 text-[11px] leading-relaxed text-slate-400">
                        {source.customer_query}
                      </p>
                      <div className="mt-2 border-t border-white/[0.05] pt-2">
                        <p className="text-[9px] uppercase tracking-wider text-slate-600">
                          Previous response
                        </p>
                        <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-slate-500">
                          {source.agent_response}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-4 rounded-lg border border-dashed border-white/[0.08] p-4 text-center">
                  <p className="text-[13px] text-slate-600">
                    Retrieved support conversations will appear here.
                  </p>
                </div>
              )}
            </section>
          </aside>
        </div>

        {/* Error toast */}
        {error && (
          <div className="absolute bottom-20 left-1/2 z-50 w-[calc(100%-2rem)] max-w-md -translate-x-1/2 rounded-xl border border-red-500/30 bg-red-950/95 px-4 py-3 text-[13px] text-red-400 shadow-xl backdrop-blur-sm lg:bottom-6">
            <span className="font-semibold">Request failed:</span> {error}
          </div>
        )}
      </div>
    </main>
  )
}
