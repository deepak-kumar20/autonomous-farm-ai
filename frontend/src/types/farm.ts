export type FieldStateType = 'EMPTY' | 'GROWING' | 'MATURE';

export interface FieldState {
  id: number;
  state: FieldStateType;
  crop: string | null;
  progress: number;
}

export interface FarmState {
  day: number;
  cash: number;
  debt: number;
  fields: FieldState[];
  inventory: Record<string, number>;
  market_prices: Record<string, number>;
  risk_forecasts: {
    drought?: string;
    pest?: string;
    price_shock?: string;
    [key: string]: string | undefined;
  };
  /** Daily probabilities for the next unresolved loss stage (simulated, not real forecasts). */
  risks?: Record<string, number>;
  scenario_id?: string | null;
  events?: { day?: number; drought?: boolean; pest_field?: number | null; price_shock?: number | null };
}

export type ActionType = 'HOLD' | 'PLANT' | 'HARVEST' | 'SELL' | 'BUY_INPUT';

export interface ActionItem {
  id: number;
  type: ActionType;
  label: string;
  valid: boolean;
  structurally_valid?: boolean;
  reason?: string | null;
  code?: string | null;
  completion_probability?: number | null;
  field_id?: number | null;
  crop_name?: string | null;
}

export interface FarmMetrics {
  session_id: string;
  current_day: number;
  cash: number;
  debt: number;
  inventory_value: number;
  total_wealth: number;
  profit: number;
  survived: boolean;
  season_over: boolean;
  fields_planted: number;
  fields_mature: number;
}

export interface ResetResponse {
  session_id: string;
  state: FarmState;
  info: Record<string, unknown>;
}

export interface FeasibilityResult {
  accepted: boolean;
  code: string;
  reason: string;
  mode: string;
  completion_probability?: number | null;
  threshold?: number | null;
}

export interface StepResponse {
  session_id: string;
  accepted: boolean;
  feasibility: FeasibilityResult;
  state: FarmState;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: {
    final_cash?: number;
    inventory_value?: number;
    total_wealth?: number;
    profit?: number;
    survived?: boolean;
    [key: string]: unknown;
  };
}

export interface PPOActionDetails {
  id: number;
  type: string;
  label: string;
  field_id?: number | null;
  crop_name?: string | null;
}

export interface PPOPredictResponse {
  session_id: string;
  action: number;
  action_details: PPOActionDetails;
  deterministic: boolean;
  policy_top_action?: PPOActionDetails;
  filter_rejections?: number;
}

export interface PPOStepResponse {
  session_id: string;
  filter_rejections?: number;
  action: number;
  action_details: PPOActionDetails;
  state: FarmState;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  ppo_model_loaded: boolean;
}

export interface ActionHistoryEntry {
  id: string;
  day: number;
  actor: 'HUMAN' | 'PPO AI';
  actionId: number;
  actionLabel: string;
  actionType: string;
  reward: number;
  time: string;
}

export type Actor = 'HUMAN' | 'PPO AI';

/** A visual/narrative event derived by diffing the farm state before and after one step. */
export interface FarmEvent {
  id: number;
  actor: Actor;
  actionId: number;
  actionType: string;
  actionLabel: string;
  dayFrom: number;
  dayTo: number;
  reward: number;
  invalid: boolean;
  planted?: { fieldId: number; crop: string; cost: number };
  harvested?: { fieldId: number; crop: string; qty: number };
  sold?: { crop: string; qty: number; revenue: number };
  pestHits: { fieldId: number; crop: string; ratio: number }[];
  droughtFields: number[];
  priceShock: 'UP' | 'DOWN' | null;
}

export interface DaySnapshot {
  day: number;
  cash: number;
  wealth: number;
  prices: Record<string, number>;
  risks: Record<string, string | undefined>;
}

// ── Research framework: scenarios and Human-vs-AI matches ─────────────

export interface RiskDay {
  day: number;
  drought_risk: number;
  pest_risk: number;
  price_shock_risk: number;
  drought_label: string;
  pest_label: string;
  price_shock_label: string;
}

export interface ScenarioPublic {
  scenario_id: string;
  profile: string;
  seed: number;
  days: number;
  description: string;
  base_rates: { drought: number; pest: number; price_shock: number };
  starting_prices: Record<string, number>;
  timeline: RiskDay[];
  note: string;
  config?: SimConfig;
  match_id?: string | null;
}

export interface ScenarioProfile {
  name: string;
  description: string;
  drought_base: number;
  pest_base: number;
  shock_base: number;
}

export interface SimConfig {
  decision_order: string;
  filter_mode: string;
  threshold_rule: string;
  completion_threshold: number;
  market_model: string;
}

export interface DayRecord {
  day: number;
  action_id: number;
  action_type: string;
  action_label: string;
  action_valid: boolean;
  invalid_reason?: string | null;
  feasibility_rejections: number;
  rejection_codes: string;
  drought_event: boolean;
  pest_event: boolean;
  price_shock_event: boolean;
  pest_destroyed_field: number | null;
  harvest_qty: number;
  harvest_crop: string | null;
  sold_qty: number;
  sold_crop: string | null;
  revenue: number;
  planting_cost: number;
  planted_crop: string | null;
  crops_destroyed: number;
  cash_end: number;
  wealth_end: number;
  reward: number;
}

export interface EpisodeSummary {
  scenario_id: string;
  initial_wealth: number;
  final_wealth: number;
  final_cash: number;
  final_inventory_value: number;
  profit: number;
  total_revenue: number;
  total_costs: number;
  harvest_qty: number;
  crops_planted: number;
  crops_harvested: number;
  crops_destroyed: number;
  actions_attempted: number;
  invalid_actions: number;
  feasibility_rejections: number;
  candidates_evaluated: number;
  season_completed: boolean;
}

export interface AIRun {
  run_id: string;
  agent: string;
  deterministic: boolean;
  created_at: string;
  records: DayRecord[];
  summary: EpisodeSummary;
}

export interface MatchRecord {
  match_id: string;
  created_at: string;
  player_label: string;
  status: 'human_playing' | 'human_complete' | 'compared';
  scenario_id: string;
  config: SimConfig;
  human: { session_id: string; records: DayRecord[]; summary: EpisodeSummary | null };
  ai_runs: AIRun[];
}

export interface MatchListItem {
  match_id: string;
  scenario_id: string;
  created_at: string;
  player_label: string;
  status: string;
  human_profit: number | null;
  ai_runs: number;
}

export interface AgentInfo {
  agent: string;
  available: boolean;
  status: string | null;
  timesteps?: number | null;
}
