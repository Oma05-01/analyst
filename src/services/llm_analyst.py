import logging
import os
import re
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
MAX_ATTEMPTS = 2
MAX_OUTPUT_TOKENS = 700

PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
TWO_DECIMAL_RE = re.compile(r"(?<![\d.])(\d+\.\d{2})(?![\d.])")

class PreviewError(RuntimeError):
    """The LLM could not produce a usable preview (callers should retry later)."""

class LLMAnalyst:
    model = MODEL_NAME

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key, timeout=30.0, max_retries=2) if api_key else None

    @property
    def available(self):
        return self.client is not None

    @staticmethod
    def fallback_preview(home_team, away_team, p_home, p_draw, p_away, exp_home_goals, exp_away_goals):
        return (
            f"{home_team} vs {away_team}: the model gives {p_home * 100:.1f}% (home win), "
            f"{p_draw * 100:.1f}% (draw) and {p_away * 100:.1f}% (away win), with expected goals "
            f"of {exp_home_goals:.2f} to {exp_away_goals:.2f}."
        )

    @staticmethod
    def numbers_consistent(text, p_home, p_draw, p_away, exp_home_goals, exp_away_goals):
        allowed_pct = set()
        for p in (p_home, p_draw, p_away):
            allowed_pct.add(round(p * 100, 1))
            allowed_pct.add(float(round(p * 100)))
        allowed_xg = {round(exp_home_goals, 2), round(exp_away_goals, 2)}

        for raw in PERCENT_RE.findall(text):
            if round(float(raw), 1) not in allowed_pct:
                return False
        for raw in TWO_DECIMAL_RE.findall(text):
            if round(float(raw), 2) not in allowed_xg:
                return False
        return True

    def generate_match_preview(self, home_team, away_team, p_home, p_draw, p_away,
                               exp_home_goals, exp_away_goals):
        if not self.client:
            raise PreviewError("GROQ_API_KEY is not set")

        prompt = f"""
        You are an expert Premier League sports analyst. Provide a 3-bullet match insight breakdown based STRICTLY on these Dixon-Coles model outputs.

        Match: {home_team} vs {away_team}
        - {home_team} Win: {p_home * 100:.1f}%
        - Draw: {p_draw * 100:.1f}%
        - {away_team} Win: {p_away * 100:.1f}%
        - {home_team} Expected Goals: {exp_home_goals:.2f}
        - {away_team} Expected Goals: {exp_away_goals:.2f}

        RULES:
        1. Quote the provided percentages and expected goals EXACTLY as written above.
        2. DO NOT calculate differences, ratios, or new percentages (e.g., do not write "a 50% chance" or "a 1.20 gap").
        3. Do not include any other numbers with two decimal places.
        4. Analyze the tactical implications of these exact numbers without mentioning specific players, injuries, or past form.
        5. Keep the response under 120 words.
        """

        last_problem = "no attempt made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a precise, data-driven football tactical analyst. Reasoning effort: low. You follow numerical constraints perfectly without calculating unprovided numbers."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=MODEL_NAME,
                    temperature=0.1,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
            except Exception as exc:
                raise PreviewError(f"Groq API call failed: {exc}") from exc

            content = (response.choices[0].message.content or "").strip()
            
            # Optionally strip reasoning blocks if the model returns <think>...</think> tags in the final string
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            if not content:
                last_problem = "empty response"
            elif not self.numbers_consistent(content, p_home, p_draw, p_away,
                                             exp_home_goals, exp_away_goals):
                last_problem = "response contained numbers that were not supplied"
            else:
                return content
            logging.warning("Preview attempt %d/%d rejected: %s", attempt, MAX_ATTEMPTS, last_problem)

        raise PreviewError(f"No usable preview after {MAX_ATTEMPTS} attempts: {last_problem}")

    def generate_chat_reply(self, messages, forecast_context):
        """Answer follow-up questions using the selected forecast as context."""
        
        if not self.client:
            raise PreviewError("GROQ_API_KEY is not set")

        system_prompt = """
            You're Premier Quant's football analyst, chatting with a fellow football fan.

            STYLE:
            - Sound relaxed, sharp, and natural. Don't sound like a report.
            - Answer the exact question first.
            - Usually use 2–4 sentences and stay under 90 words.
            - Use plain text. No Markdown, headings, tables, bold text, or numbered lists.
            - Don't repeat every figure when only one or two matter.
            - Explain the reasoning, but stop once the question is answered.
            - Don't force slang, jokes, or football banter.

            ACCURACY:
            - Treat the supplied forecast context as the source of truth.
            - Quote its numerical figures exactly as supplied. Don't invent,
            recalculate, or substitute different figures.
            - Clearly distinguish the baseline from the lineup-adjusted forecast.
            - Lineup deltas represent lineup-derived attacking XGI/90 features.
            They do not directly measure overall player quality, chemistry,
            defensive strength, or clean-sheet ability.
            - Never invent injuries, player news, recent form, or tactical details.
            - Don't claim that the experimental model is more accurate than the
            baseline without supporting evaluation evidence.
            - If the context doesn't contain the answer, say so plainly.

            Keep the conversation going naturally. Don't turn a simple question
            into a full match report.
            """
        context_message = (
            "Use this authoritative forecast context for the current conversation. "
            "It contains model outputs and available lineup-derived features; "
            "it does not provide a verified tactical scouting report.\n\n"
            + json.dumps(
                forecast_context,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
        ]

        for message in messages[-12:]:
            role = message.get("role")
            content = str(message.get("content", "")).strip()

            if role in {"user", "assistant"} and content:
                request_messages.append({
                    "role": role,
                    "content": content,
                })

        if not request_messages or request_messages[-1]["role"] != "user":
            raise PreviewError("The latest chat message must be a user question")

        try:
            response = self.client.chat.completions.create(
                messages=request_messages,
                model=MODEL_NAME,
                temperature=0.3,
                max_tokens=350,
            )
        except Exception as exc:
            raise PreviewError("Groq chat request failed") from exc

        content = (response.choices[0].message.content or "").strip()
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()

        if not content:
            raise PreviewError("Groq returned an empty chat response")

        return content