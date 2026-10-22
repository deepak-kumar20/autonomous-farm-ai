import React from 'react';
import { Info } from 'lucide-react';
import type { RiskDay } from '../../types/farm';

/* ─── 30-day risk calendar ─────────────────────────────────────────── */
export const RiskCalendar: React.FC<{ timeline: RiskDay[] }> = ({ timeline }) => {
  const rows: { key: keyof RiskDay; label: string }[] = [
    { key: 'drought_risk', label: 'Drought' },
    { key: 'pest_risk', label: 'Pests' },
    { key: 'price_shock_risk', label: 'Price shock' },
  ];
  // Sequential single hue: light → dark with probability
  const shade = (p: number) => {
    const t = Math.min(1, p / 0.35);
    const l = 95 - t * 55;
    return `hsl(24 85% ${l}%)`;
  };
  return (
    <div>
      <div className="overflow-x-auto no-scrollbar">
        <table className="border-separate border-spacing-[3px] text-[10px] font-bold text-[#6b4f36]">
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <th className="text-left pr-2 whitespace-nowrap text-xs font-extrabold text-[#3b2412]">{r.label}</th>
                {timeline.map((d) => {
                  const p = d[r.key] as number;
                  return (
                    <td
                      key={d.day}
                      title={`Day ${d.day}: ${(p * 100).toFixed(1)}% chance`}
                      className="w-6 h-6 min-w-6 rounded-md text-center align-middle"
                      style={{ background: shade(p), color: p > 0.2 ? '#fff' : '#6b4f36' }}
                    >
                      {Math.round(p * 100)}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr>
              <th />
              {timeline.map((d) => (
                <td key={d.day} className="text-center text-[#9a6a3c]">
                  {d.day}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] font-bold text-[#6b4f36]">
        <span>Daily chance (%):</span>
        {[0.02, 0.08, 0.15, 0.25, 0.35].map((p) => (
          <span key={p} className="flex items-center gap-1">
            <span className="w-4 h-4 rounded" style={{ background: shade(p) }} />
            {Math.round(p * 100)}
          </span>
        ))}
        <span className="ml-auto flex items-center gap-1 text-[#9a6a3c]">
          <Info className="w-3.5 h-3.5" /> Simulated risks, not forecasts and not outcomes. Whether an event happens is decided by the scenario's hidden draws.
        </span>
      </div>
    </div>
  );
};

