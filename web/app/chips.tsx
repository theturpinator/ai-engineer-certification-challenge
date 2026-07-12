// Five-stat delta chips ("PWR +15") shared by the chat sponsored cards and
// the garage Upgrade Shop. Zero deltas render nothing.

export type Deltas = Partial<Record<string, number>>;

const STAT_LABELS: [string, string][] = [
  ["power", "PWR"],
  ["acceleration", "ACC"],
  ["top_speed", "SPD"],
  ["handling", "HDL"],
  ["braking", "BRK"],
];

export function DeltaChips({ deltas }: { deltas?: Deltas | null }) {
  if (!deltas) return null;
  const nonZero = STAT_LABELS.filter(([k]) => deltas[k]);
  if (!nonZero.length) return null;
  return (
    <div className="deltachips">
      {nonZero.map(([k, label]) => {
        const v = deltas[k] ?? 0;
        return (
          <span key={k} className={`deltachip ${v > 0 ? "up" : "down"}`}>
            {label} {v > 0 ? `+${v}` : v}
          </span>
        );
      })}
    </div>
  );
}
