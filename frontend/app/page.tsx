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

  const isHuman = response?.escalate === true

  return (
    <main className="min-h-screen bg-[#070b14] text-white">
      {/* ===================================================== */}
      {/* SIDEBAR */}
      {/* ===================================================== */}

      <div className="flex min-h-screen">
        <aside className="hidden w-64 shrink-0 border-r border-slate-800/80 bg-[#0b101c] lg:flex lg:flex-col">
          {/* Logo */}
          <div className="border-b border-slate-800/80 px-6 py-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-lg font-black text-slate-950">
                S
              </div>

              <div>
                <h1 className="text-lg font-bold tracking-tight">SupportIQ</h1>

                <p className="text-xs text-slate-500">Support Intelligence</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6">
            <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">
              Workspace
            </p>

            <div className="space-y-1">
              <button className="flex w-full items-center gap-3 rounded-xl bg-slate-800/80 px-3 py-3 text-sm font-medium text-white">
                <span className="text-base">▣</span>
                Inbox
                <span className="ml-auto rounded-md bg-white px-2 py-0.5 text-[10px] font-bold text-slate-950">
                  1
                </span>
              </button>

              <button className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm text-slate-400 transition hover:bg-slate-800/50 hover:text-white">
                <span>◉</span>
                Conversations
              </button>

              <button className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm text-slate-400 transition hover:bg-slate-800/50 hover:text-white">
                <span>◌</span>
                Analytics
              </button>

              <button className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm text-slate-400 transition hover:bg-slate-800/50 hover:text-white">
                <span>◇</span>
                Knowledge Base
              </button>
            </div>

            <p className="mb-3 mt-10 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">
              System
            </p>

            <div className="space-y-1">
              <button className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm text-slate-400 transition hover:bg-slate-800/50 hover:text-white">
                <span>⚙</span>
                Settings
              </button>

              <button className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm text-slate-400 transition hover:bg-slate-800/50 hover:text-white">
                <span>?</span>
                Help
              </button>
            </div>
          </nav>

          {/* System status */}
          <div className="border-t border-slate-800/80 p-4">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-green-400" />
                <span className="text-xs font-medium text-slate-300">
                  System operational
                </span>
              </div>

              <p className="mt-2 text-[11px] leading-5 text-slate-500">
                Retrieval and escalation services are running normally.
              </p>
            </div>
          </div>
        </aside>

        {/* ===================================================== */}
        {/* MAIN AREA */}
        {/* ===================================================== */}

        <section className="flex min-w-0 flex-1 flex-col">
          {/* Header */}
          <header className="flex h-20 items-center justify-between border-b border-slate-800/80 bg-[#090e19]/90 px-6 lg:px-8">
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
                Agent Workspace
              </p>

              <h2 className="mt-1 text-lg font-semibold">
                Customer Support Inbox
              </h2>
            </div>

            <div className="flex items-center gap-4">
              <div className="hidden text-right sm:block">
                <p className="text-sm font-medium">AI Support Agent</p>

                <p className="text-xs text-slate-500">AmazonHelp</p>
              </div>

              <div className="flex h-10 w-10 items-center justify-center rounded-full border border-slate-700 bg-slate-800 text-sm font-semibold">
                AI
              </div>
            </div>
          </header>

          {/* Content */}
          <div className="flex-1 overflow-auto">
            <div className="mx-auto max-w-[1500px] p-5 lg:p-8">
              {/* Top stats */}
              <div className="mb-6 grid gap-4 sm:grid-cols-3">
                <div className="rounded-2xl border border-slate-800 bg-[#0d1422] p-5">
                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Processing Mode
                  </p>

                  <div className="mt-3 flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-green-400" />
                    <span className="text-lg font-semibold">RAG Active</span>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-[#0d1422] p-5">
                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Knowledge Sources
                  </p>

                  <p className="mt-3 text-lg font-semibold">148,556</p>

                  <p className="mt-1 text-xs text-slate-500">
                    Support conversations
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-[#0d1422] p-5">
                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Retrieval
                  </p>

                  <p className="mt-3 text-lg font-semibold">FAISS</p>

                  <p className="mt-1 text-xs text-slate-500">
                    Top-5 semantic search
                  </p>
                </div>
              </div>

              {/* Main dashboard */}
              <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_390px]">
                {/* ================================================= */}
                {/* CONVERSATION */}
                {/* ================================================= */}

                <section className="flex min-h-[650px] flex-col overflow-hidden rounded-2xl border border-slate-800 bg-[#0d1422]">
                  {/* Conversation header */}
                  <div className="flex items-center justify-between border-b border-slate-800 px-6 py-5">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-800 text-sm">
                        C
                      </div>

                      <div>
                        <p className="text-sm font-semibold">
                          Customer Conversation
                        </p>

                        <p className="text-xs text-slate-500">
                          Live AI-assisted support
                        </p>
                      </div>
                    </div>

                    <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-[11px] text-slate-400">
                      New conversation
                    </span>
                  </div>

                  {/* Messages */}
                  <div className="flex-1 overflow-auto p-6">
                    {!response && !loading && (
                      <div className="flex min-h-[430px] flex-col items-center justify-center text-center">
                        <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 text-2xl">
                          ✦
                        </div>

                        <h3 className="text-lg font-semibold">
                          Ready to assist
                        </h3>

                        <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                          Enter a customer question below. SupportIQ will
                          retrieve relevant historical conversations, assess
                          risk, and decide whether AI or a human agent should
                          handle the request.
                        </p>
                      </div>
                    )}

                    {loading && (
                      <div className="flex min-h-[430px] flex-col items-center justify-center">
                        <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-white" />

                        <p className="mt-4 text-sm text-slate-400">
                          Analyzing customer request...
                        </p>

                        <p className="mt-1 text-xs text-slate-600">
                          Retrieving relevant support conversations
                        </p>
                      </div>
                    )}

                    {response && (
                      <div className="space-y-7">
                        {/* Customer message */}
                        <div className="flex gap-4">
                          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-800 text-xs font-semibold">
                            C
                          </div>

                          <div className="max-w-3xl">
                            <div className="mb-1 flex items-center gap-2">
                              <span className="text-xs font-semibold text-slate-300">
                                Customer
                              </span>

                              <span className="text-[10px] text-slate-600">
                                Just now
                              </span>
                            </div>

                            <div className="rounded-2xl rounded-tl-sm bg-slate-800 px-5 py-4 text-sm leading-6 text-slate-200">
                              {response.query}
                            </div>
                          </div>
                        </div>

                        {/* Decision banner */}
                        <div
                          className={`rounded-2xl border p-5 ${
                            isHuman
                              ? 'border-red-900/70 bg-red-950/20'
                              : 'border-green-900/50 bg-green-950/10'
                          }`}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-4">
                            <div className="flex items-center gap-3">
                              <div
                                className={`flex h-10 w-10 items-center justify-center rounded-xl ${
                                  isHuman
                                    ? 'bg-red-500/10 text-red-400'
                                    : 'bg-green-500/10 text-green-400'
                                }`}
                              >
                                {isHuman ? '!' : '✓'}
                              </div>

                              <div>
                                <p
                                  className={`text-sm font-semibold ${
                                    isHuman ? 'text-red-400' : 'text-green-400'
                                  }`}
                                >
                                  {isHuman
                                    ? 'Human escalation required'
                                    : 'AI can handle this request'}
                                </p>

                                <p className="mt-1 text-xs text-slate-500">
                                  Decision engine: {response.reason}
                                </p>
                              </div>
                            </div>

                            <div className="flex gap-2">
                              <span className="rounded-full bg-slate-800 px-3 py-1.5 text-xs text-slate-300">
                                Confidence {confidencePercent.toFixed(1)}%
                              </span>

                              <span
                                className={`rounded-full px-3 py-1.5 text-xs ${
                                  isHuman
                                    ? 'bg-red-500/10 text-red-400'
                                    : 'bg-green-500/10 text-green-400'
                                }`}
                              >
                                Risk: {response.risk_level}
                              </span>
                            </div>
                          </div>

                          {response.risk_categories.length > 0 && (
                            <div className="mt-4 border-t border-red-900/30 pt-4">
                              <p className="text-xs text-slate-500">
                                Detected risk categories
                              </p>

                              <div className="mt-2 flex flex-wrap gap-2">
                                {response.risk_categories.map((category) => (
                                  <span
                                    key={category}
                                    className="rounded-md bg-red-500/10 px-2.5 py-1 text-xs text-red-400"
                                  >
                                    {category}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* AI response */}
                        <div className="flex gap-4">
                          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white text-xs font-black text-slate-950">
                            S
                          </div>

                          <div className="max-w-3xl">
                            <div className="mb-1 flex items-center gap-2">
                              <span className="text-xs font-semibold text-slate-300">
                                SupportIQ
                              </span>

                              <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[9px] uppercase tracking-wider text-slate-500">
                                AI
                              </span>
                            </div>

                            <div className="rounded-2xl rounded-tl-sm border border-slate-800 bg-slate-900 px-5 py-4 text-sm leading-7 text-slate-300">
                              {response.answer ||
                                'This conversation should be handled by a human support agent.'}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Composer */}
                  <div className="border-t border-slate-800 p-5">
                    <div className="flex gap-3">
                      <input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            sendMessage()
                          }
                        }}
                        placeholder="Type a customer message..."
                        className="min-w-0 flex-1 rounded-xl border border-slate-700 bg-[#080d17] px-4 py-3.5 text-sm text-white outline-none placeholder:text-slate-600 transition focus:border-slate-500"
                      />

                      <button
                        onClick={sendMessage}
                        disabled={loading || !query.trim()}
                        className="rounded-xl bg-white px-6 py-3 font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {loading ? '...' : 'Send'}
                      </button>
                    </div>

                    <p className="mt-2 px-1 text-[10px] text-slate-600">
                      Press Enter to send • SupportIQ analyzes every request
                      before generating a response.
                    </p>
                  </div>
                </section>

                {/* ================================================= */}
                {/* AI ANALYSIS */}
                {/* ================================================= */}

                <aside className="space-y-6">
                  {/* Confidence card */}
                  <section className="rounded-2xl border border-slate-800 bg-[#0d1422] p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs uppercase tracking-wider text-slate-500">
                          AI Analysis
                        </p>

                        <h3 className="mt-1 text-base font-semibold">
                          Retrieval Confidence
                        </h3>
                      </div>

                      <span className="text-xs text-slate-600">
                        BGE + FAISS
                      </span>
                    </div>

                    {response ? (
                      <>
                        <div className="mt-7 flex items-end justify-between">
                          <div>
                            <span className="text-5xl font-bold tracking-tight">
                              {confidencePercent.toFixed(1)}
                            </span>

                            <span className="ml-1 text-xl text-slate-500">
                              %
                            </span>
                          </div>

                          <span className="mb-2 rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-400">
                            {response.confidence_level}
                          </span>
                        </div>

                        <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-800">
                          <div
                            className="h-full rounded-full bg-white transition-all duration-500"
                            style={{
                              width: `${confidencePercent}%`,
                            }}
                          />
                        </div>

                        <div className="mt-4 grid grid-cols-2 gap-3">
                          <div className="rounded-xl bg-slate-900 p-3">
                            <p className="text-[10px] uppercase tracking-wider text-slate-600">
                              Max Similarity
                            </p>

                            <p className="mt-1 text-sm font-semibold">
                              {response.max_similarity
                                ? (response.max_similarity * 100).toFixed(1)
                                : '—'}
                              %
                            </p>
                          </div>

                          <div className="rounded-xl bg-slate-900 p-3">
                            <p className="text-[10px] uppercase tracking-wider text-slate-600">
                              Avg Similarity
                            </p>

                            <p className="mt-1 text-sm font-semibold">
                              {response.average_similarity
                                ? (response.average_similarity * 100).toFixed(1)
                                : '—'}
                              %
                            </p>
                          </div>
                        </div>
                      </>
                    ) : (
                      <p className="mt-6 text-sm text-slate-600">
                        Confidence metrics will appear after a request.
                      </p>
                    )}
                  </section>

                  {/* Decision card */}
                  <section className="rounded-2xl border border-slate-800 bg-[#0d1422] p-6">
                    <p className="text-xs uppercase tracking-wider text-slate-500">
                      Routing Decision
                    </p>

                    {response ? (
                      <div className="mt-5">
                        <div
                          className={`flex items-center gap-4 rounded-xl p-4 ${
                            isHuman ? 'bg-red-950/30' : 'bg-green-950/20'
                          }`}
                        >
                          <div
                            className={`flex h-11 w-11 items-center justify-center rounded-xl text-lg ${
                              isHuman
                                ? 'bg-red-500/10 text-red-400'
                                : 'bg-green-500/10 text-green-400'
                            }`}
                          >
                            {isHuman ? '!' : '✓'}
                          </div>

                          <div>
                            <p
                              className={`font-semibold ${
                                isHuman ? 'text-red-400' : 'text-green-400'
                              }`}
                            >
                              {isHuman ? 'Human Escalation' : 'AI Handled'}
                            </p>

                            <p className="mt-1 text-xs text-slate-500">
                              {response.reason}
                            </p>
                          </div>
                        </div>

                        <div className="mt-4 space-y-3">
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-500">Risk level</span>

                            <span className="font-medium capitalize">
                              {response.risk_level}
                            </span>
                          </div>

                          <div className="flex justify-between text-sm">
                            <span className="text-slate-500">Decision</span>

                            <span className="font-medium capitalize">
                              {response.decision}
                            </span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="mt-5 text-sm text-slate-600">
                        No routing decision yet.
                      </p>
                    )}
                  </section>

                  {/* Sources */}
                  <section className="rounded-2xl border border-slate-800 bg-[#0d1422] p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs uppercase tracking-wider text-slate-500">
                          Knowledge Retrieval
                        </p>

                        <h3 className="mt-1 font-semibold">
                          Top Support Sources
                        </h3>
                      </div>

                      {response && (
                        <span className="rounded-full bg-slate-800 px-2.5 py-1 text-[10px] text-slate-400">
                          {response.retrieved_results.length} results
                        </span>
                      )}
                    </div>

                    {response ? (
                      <div className="mt-5 space-y-3">
                        {response.retrieved_results.map((source, index) => (
                          <div
                            key={index}
                            className="rounded-xl border border-slate-800 bg-[#080d17] p-4 transition hover:border-slate-700"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
                                Source {index + 1}
                              </span>

                              <span className="text-xs font-semibold text-slate-300">
                                {(source.similarity * 100).toFixed(1)}%
                              </span>
                            </div>

                            <p className="mt-3 line-clamp-3 text-xs leading-5 text-slate-400">
                              {source.customer_query}
                            </p>

                            <div className="mt-3 border-t border-slate-800 pt-3">
                              <p className="text-[10px] uppercase tracking-wider text-slate-600">
                                Previous response
                              </p>

                              <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-500">
                                {source.agent_response}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="mt-5 rounded-xl border border-dashed border-slate-800 p-5 text-center">
                        <p className="text-sm text-slate-600">
                          Retrieved support conversations will appear here.
                        </p>
                      </div>
                    )}
                  </section>
                </aside>
              </div>

              {/* Error */}
              {error && (
                <div className="mt-6 rounded-xl border border-red-900/60 bg-red-950/20 p-4 text-sm text-red-400">
                  <span className="font-semibold">Request failed:</span> {error}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}
