-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. TEAMS
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,      -- Canonical name e.g., "Manchester United"
    short_code VARCHAR(10) NOT NULL,        -- e.g., "MUN"
    primary_colour VARCHAR(7),               -- Hex code e.g., "#DA291C"
    secondary_colour VARCHAR(7),             -- Hex code e.g., "#FFFFFF"
    aliases JSONB DEFAULT '[]'::jsonb       -- Known raw variants
);

-- 2. MATCHES
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    season VARCHAR(10) NOT NULL,             -- e.g., "2023-24"
    matchweek INT NOT NULL,
    kickoff_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team_id INT REFERENCES teams(id),
    away_team_id INT REFERENCES teams(id),
    status VARCHAR(20) DEFAULT 'SCHEDULED',  -- SCHEDULED, FINISHED, POSTPONED
    home_goals INT,
    away_goals INT,
    home_xg NUMERIC(4, 2),
    away_xg NUMERIC(4, 2),
    home_shots INT,
    away_shots INT,
    home_corners INT,
    away_corners INT,
    home_cards INT,
    away_cards INT,
    referee VARCHAR(100),
    CONSTRAINT unique_fixture UNIQUE (season, home_team_id, away_team_id)
);

-- 3. TEAM RATINGS (Historical Snapshots)
CREATE TABLE IF NOT EXISTS team_ratings (
    id SERIAL PRIMARY KEY,
    team_id INT REFERENCES teams(id),
    as_of_date TIMESTAMP WITH TIME ZONE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    attack NUMERIC(6, 4) NOT NULL,
    defence NUMERIC(6, 4) NOT NULL,
    elo NUMERIC(6, 2) NOT NULL,
    CONSTRAINT unique_rating_snapshot UNIQUE (team_id, as_of_date, model_version)
);

-- 4. PREDICTIONS (Immutable once locked)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(id),
    model_version VARCHAR(50) NOT NULL,
    p_home NUMERIC(5, 4) NOT NULL,
    p_draw NUMERIC(5, 4) NOT NULL,
    p_away NUMERIC(5, 4) NOT NULL,
    expected_home_goals NUMERIC(4, 2) NOT NULL,
    expected_away_goals NUMERIC(4, 2) NOT NULL,
    scoreline_matrix JSONB,                  -- Full score probability matrix
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    locked_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT unique_match_model_pred UNIQUE (match_id, model_version)
);

-- 5. PREDICTION SCORES
CREATE TABLE IF NOT EXISTS prediction_scores (
    id SERIAL PRIMARY KEY,
    prediction_id INT REFERENCES predictions(id) UNIQUE,
    outcome VARCHAR(10) NOT NULL,             -- 'HOME', 'DRAW', 'AWAY'
    brier NUMERIC(6, 4) NOT NULL,
    rps NUMERIC(6, 4) NOT NULL,
    log_loss NUMERIC(6, 4) NOT NULL,
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. PLAYERS
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    team_id INT REFERENCES teams(id),
    name VARCHAR(150) NOT NULL,
    position VARCHAR(20),                     -- GK, DEF, MID, FWD
    aliases JSONB DEFAULT '[]'::jsonb
);

-- 7. INJURY REPORTS
CREATE TABLE IF NOT EXISTS injury_reports (
    id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(id),
    team_id INT REFERENCES teams(id),
    status VARCHAR(50) NOT NULL,              -- out, doubtful, available
    expected_return VARCHAR(100),
    source_url TEXT,
    source_name VARCHAR(100),
    extracted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    confidence NUMERIC(3, 2) DEFAULT 1.00
);

-- 8. NEWS ITEMS
CREATE TABLE IF NOT EXISTS news_items (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100) NOT NULL,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 9. CHAT EVENTS (Minimal telemetry)
CREATE TABLE IF NOT EXISTS chat_events (
    id SERIAL PRIMARY KEY,
    user_hash VARCHAR(64),                   -- Anonymized IP hash
    query TEXT,
    tool_calls JSONB,
    tokens_used INT,
    response_time_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_matches_kickoff ON matches(kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_matches_season_mw ON matches(season, matchweek);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);