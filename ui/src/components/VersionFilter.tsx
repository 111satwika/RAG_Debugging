import type { Versions } from "../types";

export const ALL_VERSIONS = "all";

type Props = { versions: Versions | null; value: string; onChange: (version: string) => void };

/** Pipeline version filter (v1, v2, ...). The version is bumped in src/config.py after each improvement. */
export function VersionFilter({ versions, value, onChange }: Props) {
  return (
    <label className="inline-field version-filter">
      Version
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value={ALL_VERSIONS}>All versions</option>
        {versions?.versions.map((v) => (
          <option key={v} value={v}>
            {v}
            {v === versions.current ? " (current)" : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

export const versionParam = (value: string): string | undefined => (value === ALL_VERSIONS ? undefined : value);
