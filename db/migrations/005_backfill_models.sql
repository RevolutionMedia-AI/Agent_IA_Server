-- ponytail: agent model backfill to align with the streaming-only
-- catalog. Maps legacy / non-catalog values to the closest valid model
-- so existing agents don't silently break when the FE catalog trims
-- down. Idempotent (safe to re-run on every deploy).

-- STT: legacy batch models → current Realtime defaults
UPDATE agents
   SET stt_model = 'gpt-realtime'
 WHERE stt_provider = 'openai'
   AND stt_model IN ('gpt-4o-transcribe', 'whisper-1');

UPDATE agents
   SET stt_model = 'gpt-4o-mini-realtime-preview'
 WHERE stt_provider = 'openai'
   AND stt_model = 'gpt-4o-mini-transcribe';

-- STT: drop Rime entirely. The user removed Rime STT from the catalog.
UPDATE agents
   SET stt_model = NULL,
       stt_provider = 'openai'
 WHERE stt_provider = 'rime';

-- LLM OpenAI: anything outside the 3-model catalog → gpt-4o (default)
UPDATE agents
   SET llm_model = 'gpt-4o'
 WHERE llm_provider = 'openai'
   AND llm_model IS NOT NULL
   AND llm_model NOT IN ('gpt-4o', 'gpt-4o-mini', 'o4-mini');

-- LLM OpenAI: NULL → gpt-4o
UPDATE agents
   SET llm_model = 'gpt-4o'
 WHERE llm_provider = 'openai'
   AND (llm_model IS NULL OR llm_model = '');

-- LLM Anthropic: anything outside the 3-model catalog → claude-sonnet-4-5
UPDATE agents
   SET llm_model = 'claude-sonnet-4-5'
 WHERE llm_provider = 'anthropic'
   AND llm_model IS NOT NULL
   AND llm_model NOT IN ('claude-sonnet-4-5', 'claude-haiku-3-5', 'claude-haiku-3');

-- LLM Anthropic: NULL → claude-sonnet-4-5
UPDATE agents
   SET llm_model = 'claude-sonnet-4-5'
 WHERE llm_provider = 'anthropic'
   AND (llm_model IS NULL OR llm_model = '');

-- LLM Gemini: anything → gemini-2-5-flash (cheap + streaming, safe default)
UPDATE agents
   SET llm_model = 'gemini-2-5-flash'
 WHERE llm_provider = 'gemini'
   AND (llm_model IS NULL OR llm_model = ''
        OR llm_model NOT IN ('gemini-1-5-flash', 'gemini-2-5-flash',
                              'gemini-3-1-pro', 'gemini-3-1-pro-long',
                              'gemini-3-5-flash', 'gemini-3-flash',
                              'gemini-3-1-flash-lite', 'gemini-2-5-pro',
                              'gemini-2-5-pro-long', 'gemini-2-5-flash-lite',
                              'gemini-embeddings'));

-- LLM MiniMax: anything outside the 2-model catalog → MiniMax-M3
UPDATE agents
   SET llm_model = 'MiniMax-M3'
 WHERE llm_provider = 'minimax'
   AND llm_model IS NOT NULL
   AND llm_model NOT IN ('MiniMax-M3', 'MiniMax-M2.7');

UPDATE agents
   SET llm_model = 'MiniMax-M3'
 WHERE llm_provider = 'minimax'
   AND (llm_model IS NULL OR llm_model = '');
