import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { Dices, Play, Bot, Swords, RefreshCw } from 'lucide-react';
import { RiskCalendar } from '../components/game/RiskCalendar';
import { farmApi } from '../services/api';
import { useFarmContext } from '../context/FarmContext';
import { PageBanner, SectionTitle } from '../components/game/common';
import { Robot, Scarecrow } from '../components/art/Scenery';
import { formatMoney, prettyLabel } from '../game/crops';
import type { AgentInfo, DayRecord, EpisodeSummary, MatchListItem, MatchRecord, RiskDay, ScenarioProfile, ScenarioPublic } from '../types/farm';

const PLAYER_COLOR: Record<string, string> = { human: '#c98500', random: '#2a78d6', rule_based: '#eb6834', ppo: '#1baf7a' };
const PLAYER_NAME: Record<string, string> = { human: 'You', random: 'Random AI', rule_based: 'Rule-based AI', ppo: 'PPO AI' };
const FILTERS = [
  { id: 'deterministic', label: 'Deterministic filter', hint: 'Blocks impossible moves and crops that cannot finish by day 30 (recommended)' },
  { id: 'none', label: 'No filter', hint: 'Original rules: impossible moves fail and still cost a day' },
  { id: 'uncertainty', label: 'Risk-aware filter', hint: 'Also blocks plantings whose chance of finishing is below break-even' },
];

/* ─── Multi-series wealth chart ────────────────────────────────────── */
interface Series {
  id: string;
  label: string;
  color: string;
  values: number[]; // day 0..30
}

const WealthCompare: React.FC<{ series: Series[] }> = ({ series }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const W = 640, H = 240, pad = { l: 60, r: 12, t: 10, b: 24 };
  const all = series.flatMap((s) => s.values);
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = hi - lo || 1;
  const y0 = lo - span * 0.08, y1 = hi + span * 0.08;
  const xs = (d: number) => pad.l + (d / 30) * (W - pad.l - pad.r);
  const ys = (v: number) => pad.t + (1 - (v - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const ticks = [0, 0.5, 1].map((t) => y0 + (y1 - y0) * (0.1 + 0.8 * t));
  return (
    <div>
      <div className="flex flex-wrap gap-3 mb-2">
        {series.map((s) => (
          <span key={s.id} className="flex items-center gap-1.5 text-xs font-bold text-[#3b2412]">
            <span className="w-5 h-1 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <div
        ref={ref}
        className="relative"
        onPointerMove={(e) => {
          const r = ref.current?.getBoundingClientRect();
          if (!r) return;
          const vx = ((e.clientX - r.left) / r.width) * W;
          setHover(Math.max(0, Math.min(30, Math.round(((vx - pad.l) / (W - pad.l - pad.r)) * 30))));
        }}
        onPointerLeave={() => setHover(null)}
      >
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full block" style={{ height: H }} preserveAspectRatio="none" role="img" aria-label="Wealth by day for each player">
          {ticks.map((t) => (
            <line key={t} x1={pad.l} x2={W - pad.r} y1={ys(t)} y2={ys(t)} stroke="#e2c48d" vectorEffect="non-scaling-stroke" />
          ))}
          <line x1={pad.l} x2={W - pad.r} y1={ys(10000)} y2={ys(10000)} stroke="#9a6a3c" strokeDasharray="5 5" vectorEffect="non-scaling-stroke" />
          {series.map((s) => (
            <path
              key={s.id}
              d={s.values.map((v, d) => `${d ? 'L' : 'M'}${xs(d)} ${ys(v)}`).join(' ')}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
              strokeLinejoin="round"
            />
          ))}
          {hover !== null && <line x1={xs(hover)} x2={xs(hover)} y1={pad.t} y2={H - pad.b} stroke="#6b4f36" vectorEffect="non-scaling-stroke" />}
        </svg>
        {ticks.map((t) => (
          <span key={t} className="absolute left-0 -translate-y-1/2 text-[10px] font-bold text-[#9a6a3c] tabular" style={{ top: ys(t) }}>
            ₹{Math.round(t).toLocaleString('en-IN')}
          </span>
        ))}
        <span className="absolute left-[60px] bottom-0 text-[10px] font-bold text-[#9a6a3c]">Start</span>
        <span className="absolute right-3 bottom-0 text-[10px] font-bold text-[#9a6a3c]">Day 30</span>
        {hover !== null && (
          <div
            className="absolute top-1 z-10 pointer-events-none rounded-lg bg-[#3b2412] text-[#fff6df] px-2.5 py-1.5 text-xs font-bold -translate-x-1/2"
            style={{ left: `${Math.min(85, Math.max(15, (xs(hover) / W) * 100))}%` }}
          >
            <div className="opacity-75">{hover === 0 ? 'Start' : `End of day ${hover}`}</div>
            {series.map((s) => (
              <div key={s.id} className="flex items-center gap-1.5 tabular">
                <span className="w-2 h-2 rounded-full" style={{ background: s.color }} />
                {s.label}: {formatMoney(s.values[hover])}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const toSeries = (id: string, label: string, records: DayRecord[]): Series => ({
  id,
  label,
  color: PLAYER_COLOR[id] ?? '#6250d6',
  values: [10000, ...records.map((r) => r.wealth_end)],
});

const METRICS: { key: keyof EpisodeSummary; label: string; kind: 'money' | 'num' }[] = [
  { key: 'final_wealth', label: 'Final wealth', kind: 'money' },
  { key: 'profit', label: 'Profit', kind: 'money' },
  { key: 'total_revenue', label: 'Sales revenue', kind: 'money' },
  { key: 'total_costs', label: 'Seed costs', kind: 'money' },
  { key: 'harvest_qty', label: 'Harvest (units)', kind: 'num' },
  { key: 'crops_planted', label: 'Crops planted', kind: 'num' },
  { key: 'crops_destroyed', label: 'Crops lost to pests', kind: 'num' },
  { key: 'invalid_actions', label: 'Invalid actions', kind: 'num' },
  { key: 'feasibility_rejections', label: 'Rejected attempts', kind: 'num' },
];

function describeDay(r?: DayRecord): string {
  if (!r) return '—';
  const base = prettyLabel(r.action_label);
  if (!r.action_valid) return `${base} (failed)`;
  if (r.harvest_qty) return `Harvest ${r.harvest_qty} ${r.harvest_crop}`;
  if (r.sold_qty) return `Sell ${r.sold_qty} ${r.sold_crop} · ${formatMoney(r.revenue)}`;
  return base;
}

/* ─── Match view ───────────────────────────────────────────────────── */
const MatchView: React.FC<{ matchId: string }> = ({ matchId }) => {
  const farm = useFarmContext();
  const navigate = useNavigate();
  const [match, setMatch] = useState<MatchRecord | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [agent, setAgent] = useState('ppo');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setMatch(await farmApi.getMatch(matchId));
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Could not load match');
    }
  }, [matchId]);

  useEffect(() => {
    load();
    farmApi.listAgents().then((a) => {
      setAgents(a);
      if (!a.find((x) => x.agent === 'ppo')?.available) setAgent('rule_based');
    }).catch(() => undefined);
  }, [load]);

  if (err) return <div className="panel p-5 font-bold text-[#8f2621]">{err}</div>;
  if (!match) return <div className="panel p-5 font-bold">Loading match…</div>;

  const human = match.human;
  const runs = match.ai_runs;
  const lastRun = runs[runs.length - 1];
  const series = [
    ...(human.summary ? [toSeries('human', `You (${match.player_label})`, human.records)] : []),
    ...runs.map((r, i) => ({ ...toSeries(r.agent, PLAYER_NAME[r.agent] ?? r.agent, r.records), id: `${r.agent}-${i}` })),
  ];
  // Colours follow the player type, so later duplicate runs of one agent share a colour
  series.forEach((s) => (s.color = PLAYER_COLOR[s.id.split('-')[0]] ?? s.color));

  return (
    <div className="space-y-5">
      <div className="panel p-5 flex flex-wrap items-center gap-3">
        <div>
          <div className="text-[11px] font-extrabold uppercase tracking-wide text-[#9a6a3c]">Match {match.match_id}</div>
          <div className="font-display font-bold text-xl text-[#3b2412]">
            Scenario {match.scenario_id} · filter: {match.config.filter_mode}
          </div>
        </div>
        <span className="ml-auto rounded-full border-2 px-3 py-1 text-xs font-extrabold border-[#6b4423] bg-[#fdf0cf]">
          {match.status === 'human_playing' ? 'You are still playing' : match.status === 'human_complete' ? 'Your season is done' : 'Compared'}
        </span>
      </div>

      {match.status === 'human_playing' && (
        <div className="panel p-5 flex flex-wrap items-center gap-4">
          <Scarecrow size={60} />
          <p className="flex-1 font-semibold text-[#6b4f36]">Play all 30 days first. The AI only sees this scenario after you finish, so it cannot copy you.</p>
          <button
            className="btn btn-lg"
            onClick={async () => {
              await farm.loadSession(human.session_id, match.match_id);
              navigate('/farm');
            }}
          >
            <Play className="w-5 h-5" /> Continue my season
          </button>
        </div>
      )}

      {match.status !== 'human_playing' && (
        <div className="panel p-5">
          <SectionTitle>Run an AI on the same scenario</SectionTitle>
          <div className="flex flex-wrap items-center gap-3">
            <Robot size={56} thinking={busy} />
            <div className="flex flex-wrap gap-2">
              {agents.map((a) => (
                <button
                  key={a.agent}
                  disabled={!a.available || busy}
                  onClick={() => setAgent(a.agent)}
                  title={a.status ?? ''}
                  className={`rounded-xl border-[3px] px-3 py-2 font-display font-semibold text-sm disabled:opacity-40 ${agent === a.agent ? 'bg-[#ffd65c] border-[#3b2412]' : 'bg-[#fff6df] border-[#c99a66]'}`}
                >
                  {PLAYER_NAME[a.agent]}
                  {a.agent === 'ppo' && a.timesteps ? <span className="block text-[10px] text-[#6b4f36]">{(a.timesteps / 1e6).toFixed(1)}M training steps</span> : null}
                </button>
              ))}
            </div>
            <button
              className="btn btn-purple btn-lg ml-auto"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                setErr(null);
                try {
                  await farmApi.runAI(match.match_id, agent);
                  await load();
                } catch (e) {
                  setErr(e instanceof Error ? e.message : 'AI run failed');
                } finally {
                  setBusy(false);
                }
              }}
            >
              <Bot className={`w-5 h-5 ${busy ? 'animate-bounce' : ''}`} /> Run {PLAYER_NAME[agent]}
            </button>
          </div>
          <p className="text-xs font-semibold text-[#9a6a3c] mt-2">
            The AI starts from the same ₹10,000 farm, faces the same weather, pest and price-shock draws, and uses the same feasibility rules. Its moves are its own.
          </p>
        </div>
      )}

      {human.summary && (
        <>
          <div className="panel p-5 overflow-x-auto">
            <SectionTitle>Results</SectionTitle>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-wide text-[#9a6a3c]">
                  <th className="py-1 pr-3">Metric</th>
                  <th className="py-1 pr-3" style={{ color: PLAYER_COLOR.human }}>You</th>
                  {runs.map((r) => (
                    <th key={r.run_id} className="py-1 pr-3" style={{ color: PLAYER_COLOR[r.agent] }}>
                      {PLAYER_NAME[r.agent]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRICS.map((m) => (
                  <tr key={m.key} className="border-t-2 border-dashed border-[#e2c48d]">
                    <td className="py-1.5 pr-3 font-bold text-[#6b4f36]">{m.label}</td>
                    {[human.summary!, ...runs.map((r) => r.summary)].map((s, i) => (
                      <td key={i} className="py-1.5 pr-3 font-display font-semibold text-[#3b2412] tabular">
                        {m.kind === 'money' ? formatMoney(Number(s[m.key])) : String(s[m.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {lastRun && (
              <p className="mt-3 font-display font-bold text-lg text-[#3b2412]">
                {lastRun.summary.profit > human.summary.profit
                  ? `${PLAYER_NAME[lastRun.agent]} earned ${formatMoney(lastRun.summary.profit - human.summary.profit)} more than you.`
                  : lastRun.summary.profit < human.summary.profit
                  ? `You beat ${PLAYER_NAME[lastRun.agent]} by ${formatMoney(human.summary.profit - lastRun.summary.profit)}!`
                  : `A tie with ${PLAYER_NAME[lastRun.agent]}.`}{' '}
                <span className="text-xs text-[#9a6a3c]">One scenario is a single case, not a general verdict.</span>
              </p>
            )}
          </div>

          <div className="panel p-5">
            <SectionTitle>Wealth through the season</SectionTitle>
            <WealthCompare series={series} />
          </div>

          <div className="panel p-5 overflow-x-auto">
            <SectionTitle>Day by day</SectionTitle>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left uppercase tracking-wide text-[#9a6a3c]">
                  <th className="py-1 pr-2">Day</th>
                  <th className="py-1 pr-2">Events</th>
                  <th className="py-1 pr-2" style={{ color: PLAYER_COLOR.human }}>Your move</th>
                  {lastRun && <th className="py-1 pr-2" style={{ color: PLAYER_COLOR[lastRun.agent] }}>{PLAYER_NAME[lastRun.agent]} move</th>}
                  <th className="py-1 pr-2">Your wealth</th>
                  {lastRun && <th className="py-1 pr-2">AI wealth</th>}
                </tr>
              </thead>
              <tbody>
                {human.records.map((r, i) => {
                  const a = lastRun?.records[i];
                  const ev = [r.drought_event && 'Drought', r.pest_event && 'Pest outbreak', r.price_shock_event && 'Price shock'].filter(Boolean).join(', ');
                  return (
                    <tr key={r.day} className="border-t border-[#efdcb5]">
                      <td className="py-1 pr-2 font-bold">{r.day}</td>
                      <td className="py-1 pr-2 text-[#8f2621] font-bold">{ev || '—'}</td>
                      <td className="py-1 pr-2 font-semibold">{describeDay(r)}</td>
                      {lastRun && <td className="py-1 pr-2 font-semibold">{describeDay(a)}</td>}
                      <td className="py-1 pr-2 tabular">{formatMoney(r.wealth_end)}</td>
                      {lastRun && <td className="py-1 pr-2 tabular">{a ? formatMoney(a.wealth_end) : '—'}</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};

/* ─── Setup view ───────────────────────────────────────────────────── */
const SetupView: React.FC = () => {
  const farm = useFarmContext();
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<ScenarioProfile[]>([]);
  const [profile, setProfile] = useState('baseline');
  const [seed, setSeed] = useState<number>(() => Math.floor(Math.random() * 10000));
  const [filter, setFilter] = useState('deterministic');
  const [label, setLabel] = useState('player-1');
  const [scenario, setScenario] = useState<ScenarioPublic | null>(null);
  const [matches, setMatches] = useState<MatchListItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    farmApi.getProfiles().then(setProfiles).catch(() => undefined);
    farmApi.listMatches().then((m) => setMatches(m.reverse())).catch(() => undefined);
  }, []);

  useEffect(() => {
    let live = true;
    farmApi.createScenario(profile, seed).then((s) => live && setScenario(s)).catch((e) => live && setErr(String(e?.message ?? e)));
    return () => {
      live = false;
    };
  }, [profile, seed]);

  const start = async () => {
    if (!scenario) return;
    setBusy(true);
    setErr(null);
    try {
      const m = await farmApi.createMatch(scenario.scenario_id, label || 'player', filter, filter === 'uncertainty' ? 'break_even' : 'fixed');
      await farm.loadSession(m.session_id, m.match_id);
      navigate('/farm');
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Could not start the match');
    } finally {
      setBusy(false);
    }
  };

  const avg = useMemo(() => {
    if (!scenario) return null;
    const n = scenario.timeline.length;
    const m = (k: keyof RiskDay) => scenario.timeline.reduce((s, d) => s + (d[k] as number), 0) / n;
    return { drought: m('drought_risk'), pest: m('pest_risk'), shock: m('price_shock_risk') };
  }, [scenario]);

  return (
    <div className="space-y-5">
      <div className="panel p-5">
        <SectionTitle>1 · Choose a 30-day scenario</SectionTitle>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {profiles.map((p) => (
            <button
              key={p.name}
              onClick={() => setProfile(p.name)}
              className={`text-left rounded-2xl border-[3px] p-3 ${profile === p.name ? 'bg-[#ffd65c] border-[#3b2412] shadow-[0_3px_0_#3b2412]' : 'bg-[#fff6df] border-[#c99a66]'}`}
            >
              <div className="font-display font-bold text-lg capitalize text-[#3b2412]">{p.name}</div>
              <div className="text-xs font-semibold text-[#6b4f36]">{p.description}</div>
            </button>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-3">
          <label className="text-sm font-bold text-[#3b2412]">
            Seed
            <input
              type="number"
              min={0}
              value={seed}
              onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
              className="block mt-1 w-32 rounded-xl border-[3px] border-[#c99a66] bg-[#fffcf2] px-3 py-1.5 font-display"
            />
          </label>
          <button className="btn btn-cream" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>
            <Dices className="w-4 h-4" /> Random seed
          </button>
          {scenario && (
            <span className="text-sm font-bold text-[#6b4f36]">
              Scenario <span className="font-mono">{scenario.scenario_id}</span> — the same id always gives the same season.
            </span>
          )}
        </div>
      </div>

      {scenario && avg && (
        <motion.div className="panel p-5" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} key={scenario.scenario_id}>
          <SectionTitle>2 · What you know before day 1</SectionTitle>
          <div className="grid sm:grid-cols-4 gap-3 mb-4 text-sm">
            {[
              ['Starting cash', '₹10,000'],
              ['Fields', '5'],
              ['Crops', 'Wheat 5d · Rice 8d · Corn 6d'],
              ['Season', '30 days, 1 action per day'],
              ['Average drought risk', `${(avg.drought * 100).toFixed(1)}%/day`],
              ['Average pest risk', `${(avg.pest * 100).toFixed(1)}%/day`],
              ['Average price-shock risk', `${(avg.shock * 100).toFixed(1)}%/day`],
              ['Day-1 prices', Object.entries(scenario.starting_prices).map(([c, p]) => `${c} ₹${p.toFixed(0)}`).join(' · ')],
            ].map(([k, v]) => (
              <div key={k} className="panel-inset px-3 py-2">
                <div className="text-[10px] font-extrabold uppercase tracking-wide text-[#9a6a3c]">{k}</div>
                <div className="font-display font-semibold text-[#3b2412]">{v}</div>
              </div>
            ))}
          </div>
          <RiskCalendar timeline={scenario.timeline} />
        </motion.div>
      )}

      <div className="panel p-5">
        <SectionTitle>3 · Rules and start</SectionTitle>
        <div className="grid md:grid-cols-3 gap-3">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={`text-left rounded-2xl border-[3px] p-3 ${filter === f.id ? 'bg-[#efe9ff] border-[#4b33a6]' : 'bg-[#fff6df] border-[#c99a66]'}`}
            >
              <div className="font-display font-bold text-[#3b2412]">{f.label}</div>
              <div className="text-xs font-semibold text-[#6b4f36]">{f.hint}</div>
            </button>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-3">
          <label className="text-sm font-bold text-[#3b2412]">
            Player label (for the research log)
            <input
              value={label}
              maxLength={40}
              onChange={(e) => setLabel(e.target.value)}
              className="block mt-1 w-56 rounded-xl border-[3px] border-[#c99a66] bg-[#fffcf2] px-3 py-1.5 font-display"
            />
          </label>
          <button className="btn btn-lg ml-auto" disabled={!scenario || busy} onClick={start}>
            <Play className="w-5 h-5" /> Start my season
          </button>
        </div>
        {err && <p className="mt-3 text-sm font-bold text-[#8f2621]">{err}</p>}
      </div>

      {matches.length > 0 && (
        <div className="panel p-5">
          <SectionTitle right={<button className="btn btn-cream btn-sm" onClick={() => farmApi.listMatches().then((m) => setMatches(m.reverse()))}><RefreshCw className="w-3.5 h-3.5" /> Refresh</button>}>
            Saved matches
          </SectionTitle>
          <ul className="space-y-1.5">
            {matches.slice(0, 12).map((m) => (
              <li key={m.match_id} className="panel-inset flex flex-wrap items-center gap-2 px-3 py-2 text-sm">
                <span className="font-mono text-xs">{m.match_id}</span>
                <span className="font-bold text-[#3b2412]">{m.scenario_id}</span>
                <span className="text-xs text-[#9a6a3c]">{m.player_label}</span>
                <span className="text-xs font-bold text-[#6b4f36]">{m.human_profit != null ? `your profit ${formatMoney(m.human_profit)}` : 'in progress'}</span>
                <Link className="ml-auto btn btn-sm btn-sky" to={`/versus?match=${m.match_id}`}>
                  Open
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export const VersusPage: React.FC = () => {
  const [params] = useSearchParams();
  const matchId = params.get('match');
  return (
    <div>
      <PageBanner
        title="Human vs AI"
        subtitle="Same scenario, same starting farm, same rules — you play first, then the AI replays it"
        art={
          <div className="flex items-end gap-2">
            <Scarecrow size={80} />
            <Swords className="w-10 h-10 text-[#3b2412] mb-8" />
            <Robot size={100} mood="happy" />
          </div>
        }
      >
        {matchId && (
          <Link to="/versus" className="btn btn-cream btn-sm">
            New match
          </Link>
        )}
      </PageBanner>
      {matchId ? <MatchView matchId={matchId} /> : <SetupView />}
    </div>
  );
};
