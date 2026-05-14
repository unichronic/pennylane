import re


class LocalResponse:
    def __init__(self, content):
        self.content = content


class LocalTradingLLM:
    def with_structured_output(self, schema):
        return LocalStructuredLLM(self, schema)

    def invoke(self, prompt):
        text = str(prompt)
        prompt_head = text[:400]
        if "Trend and Asymmetry Specialist" in prompt_head:
            return LocalResponse(self._specialist(text, "trend"))
        if "Quality and Macro Specialist" in prompt_head:
            return LocalResponse(self._specialist(text, "quality"))
        if "Event and Counterargument Specialist" in prompt_head:
            return LocalResponse(self._specialist(text, "event"))
        if "You are a Bull Analyst" in prompt_head or "Bull Analyst here." in prompt_head:
            return LocalResponse(self._bull(text))
        if "You are a Bear Analyst" in prompt_head or "Bear Analyst here." in prompt_head:
            return LocalResponse(self._bear(text))
        if "Aggressive Risk Analyst" in prompt_head:
            return LocalResponse(
                "The upside case is credible here, provided the position is sized appropriately and strict stop controls remain in place."
            )
        if "Conservative Risk Analyst" in prompt_head:
            return LocalResponse(
                "The decision should be constrained because excessive exposure and volatility could impair capital if the trade moves against us."
            )
        if "Neutral Risk Analyst" in prompt_head:
            return LocalResponse(
                "A balanced adjustment makes the most sense: allow the trade only if risk controls reduce the effective position size."
            )
        if "Research Manager" in prompt_head:
            return LocalResponse(self._manager(text))
        return LocalResponse("Hold until the structured evidence becomes clearer.")

    def _evidence_ids(self, text, prefix=None):
        ids = []
        for match in re.findall(r"\[([A-Za-z0-9:_-]+)\]", text):
            if prefix and not match.startswith(prefix):
                continue
            if match not in ids:
                ids.append(match)
        return ids

    def _specialist(self, text, kind):
        market_ids = self._evidence_ids(text, "market:")
        fund_ids = self._evidence_ids(text, "fundamentals:")
        news_ids = self._evidence_ids(text, "news:")
        sentiment_ids = self._evidence_ids(text, "sentiment:")
        if kind == "trend":
            chosen = (market_ids or news_ids or ["market:report:1"])[:2]
            return "\n".join([
                f"- Trend/Momentum: favors longs when price action is constructive [{', '.join(chosen)}]",
                f"- Downside Asymmetry: mixed; use stop discipline because reversals can still bite [{', '.join(chosen)}]",
            ])
        if kind == "quality":
            chosen = (fund_ids or news_ids or ["fundamentals:report:1"])[:2]
            return "\n".join([
                f"- Valuation/Quality: mixed to constructive when profitability and balance-sheet quality hold up [{', '.join(chosen)}]",
                f"- Macro Sensitivity: mixed because regime changes can still pressure the thesis [{', '.join(chosen)}]",
            ])
        chosen = (news_ids or sentiment_ids or market_ids or ["news:report:1"])[:2]
        return "\n".join([
            f"- News/Event Risk: mixed; headlines can change conviction quickly [{', '.join(chosen)}]",
            f"- Counterargument Seed: the opposite side can argue crowding and reversal risk even in a constructive tape [{', '.join(chosen)}]",
        ])

    def _bull(self, text):
        bits = []
        if "trend=bullish" in text:
            bits.append("The market report already points to a constructive technical setup with bullish characteristics.")
        elif "trend=bearish" in text:
            bits.append("Even with a bearish stretch, the setup may still offer a rebound opportunity if risk is managed carefully.")
        else:
            bits.append("The setup is mixed rather than one-sided, but a measured long case can still be justified if confirmation improves.")
        if "macd_hist_positive=True" in text:
            bits.append("Positive MACD momentum strengthens the case that trend and growth expectations are improving.")
        if "rsi_extreme=True" in text:
            bits.append("RSI strength confirms demand, though it still argues for disciplined risk controls.")
        bits.append("The bear case raises valid caution points, but it argues more for sizing discipline than for rejecting the trade outright.")
        return " ".join(bits)

    def _bear(self, text):
        bits = []
        if "trend=bearish" in text:
            bits.append("The market report supports caution because the prevailing trend remains negative.")
        elif "trend=bullish" in text:
            bits.append("The bullish setup may already be crowded after a strong run, which raises reversal risk.")
        else:
            bits.append("Mixed signals make it easy to overstate conviction and underprice downside risk.")
        if "rsi_overbought=True" in text:
            bits.append("RSI is overbought, which raises reversal risk.")
        if "macd_hist_negative=True" in text:
            bits.append("Negative MACD momentum weakens the long thesis and reduces follow-through confidence.")
        bits.append("The bull case would still require strict risk limits because technical confirmation can lag fast market reversals.")
        return " ".join(bits)

    def _manager(self, text):
        if "trend=bullish" in text and "Bull Analyst:" in text:
            return "Buy"
        if "trend=bearish" in text and "Bear Analyst:" in text:
            return "Sell"
        return "Hold"


class LocalStructuredLLM:
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, prompt):
        from agents.schemas import (
            DebateArgument,
            DebateEvidence,
            DebateScorecard,
            PortfolioDecision,
            PortfolioRating,
            ResearchPlan,
            RubricDimension,
            TraderAction,
            TraderProposal,
        )

        text = str(prompt)
        evidence_ids = self.llm._evidence_ids(text)
        market_ids = self.llm._evidence_ids(text, "market:")
        fund_ids = self.llm._evidence_ids(text, "fundamentals:")
        news_ids = self.llm._evidence_ids(text, "news:")
        sentiment_ids = self.llm._evidence_ids(text, "sentiment:")

        def ids(*groups):
            out = []
            for group in groups:
                for item in group:
                    if item not in out:
                        out.append(item)
            return out[:3]

        def dim(score, confidence, rationale, *evidence_groups):
            return RubricDimension(
                support_score=score,
                confidence=confidence,
                rationale=rationale,
                evidence_ids=ids(*evidence_groups),
            )

        if self.schema is DebateArgument:
            bullish = "You are a Bull Analyst" in text
            bear_text = not bullish
            if bullish:
                thesis = "The long case is supported when the market setup and supporting reports retain enough validated confirmation."
                trend_score = 8 if "trend=bullish" in text else 5
                quality_score = 7 if fund_ids else 5
                event_score = 6 if news_ids or sentiment_ids else 4
                macro_score = 5
                asym_score = 6
                counter_score = 7 if "Last bear argument:" in text and "Last bear argument: " not in text else 5
                claims = [
                    DebateEvidence(
                        evidence_id=(market_ids or evidence_ids or ["market:report:1"])[0],
                        claim="Momentum and price structure are constructive enough to support a long case.",
                        why_it_matters="That increases the probability of upside follow-through.",
                    ),
                    DebateEvidence(
                        evidence_id=(fund_ids or evidence_ids or ["fundamentals:report:1"])[0],
                        claim="Quality and balance-sheet evidence do not contradict taking measured exposure.",
                        why_it_matters="That lowers the odds of the thesis depending only on short-term noise.",
                    ),
                    DebateEvidence(
                        evidence_id=(news_ids or sentiment_ids or evidence_ids or ["news:report:1"])[0],
                        claim="Event and sentiment evidence do not clearly invalidate the long setup.",
                        why_it_matters="That keeps the downside from being dominated by fresh negative catalysts.",
                    ),
                ]
                return DebateArgument(
                    thesis=thesis,
                    scorecard=DebateScorecard(
                        trend_momentum=dim(trend_score, 0.72, "Momentum is the strongest part of the bull case.", market_ids, news_ids),
                        valuation_quality=dim(quality_score, 0.61, "Quality supports the thesis more than it undermines it.", fund_ids),
                        news_event_risk=dim(event_score, 0.56, "Event risk is present but not thesis-breaking.", news_ids, sentiment_ids),
                        macro_sensitivity=dim(macro_score, 0.5, "Macro sensitivity is manageable rather than absent.", news_ids, fund_ids),
                        downside_asymmetry=dim(asym_score, 0.58, "The downside can be managed with sizing and stops.", market_ids, news_ids),
                        counterargument_strength=dim(counter_score, 0.57, "The bear pushback is real but not decisive.", news_ids, market_ids),
                    ),
                    key_claims=claims,
                    counterargument_response="The bear case highlights real risks, but the validated evidence still supports a measured long bias.",
                    overall_conviction=0.71 if "trend=bullish" in text else 0.58,
                )
            thesis = "The short or avoid case is stronger when reversal risk and negative evidence outweigh the upside narrative."
            trend_score = 8 if "trend=bearish" in text else 6
            quality_score = 6 if fund_ids else 5
            event_score = 7 if news_ids else 5
            macro_score = 6
            asym_score = 7
            counter_score = 7 if "Last bull argument:" in text and "Last bull argument: " not in text else 5
            claims = [
                DebateEvidence(
                    evidence_id=(market_ids or evidence_ids or ["market:report:1"])[0],
                    claim="Trend and momentum evidence leave room for downside or at least justify avoiding new long exposure.",
                    why_it_matters="Weak or crowded setups can reverse faster than optimistic narratives adjust.",
                ),
                DebateEvidence(
                    evidence_id=(news_ids or sentiment_ids or evidence_ids or ["news:report:1"])[0],
                    claim="Recent event flow leaves enough uncertainty to keep the downside case alive.",
                    why_it_matters="Fresh catalysts often dominate valuation arguments over short decision windows.",
                ),
                DebateEvidence(
                    evidence_id=(fund_ids or evidence_ids or ["fundamentals:report:1"])[0],
                    claim="Quality evidence is not strong enough to dismiss risk concentration and macro drag.",
                    why_it_matters="That weakens the margin of safety behind the bull thesis.",
                ),
            ]
            return DebateArgument(
                thesis=thesis,
                scorecard=DebateScorecard(
                    trend_momentum=dim(trend_score, 0.72, "Trend risk keeps the bear case credible.", market_ids, news_ids),
                    valuation_quality=dim(quality_score, 0.55, "Quality does not fully neutralize the risk case.", fund_ids),
                    news_event_risk=dim(event_score, 0.61, "Event risk still leans toward caution.", news_ids, sentiment_ids),
                    macro_sensitivity=dim(macro_score, 0.52, "Macro sensitivity can still pressure the thesis.", news_ids, fund_ids),
                    downside_asymmetry=dim(asym_score, 0.6, "Downside asymmetry argues for caution on entries.", market_ids, news_ids),
                    counterargument_strength=dim(counter_score, 0.58, "The bear side can still challenge optimistic assumptions.", market_ids, news_ids),
                ),
                key_claims=claims,
                counterargument_response="The bull case has some support, but it still depends on optimistic assumptions that are not fully validated.",
                overall_conviction=0.7 if "trend=bearish" in text else 0.6,
            )

        if self.schema is ResearchPlan:
            bull_ids = re.search(r"Bull validated evidence IDs:\s*(.*)", text)
            bear_ids = re.search(r"Bear validated evidence IDs:\s*(.*)", text)
            bull_supported = [item.strip() for item in (bull_ids.group(1) if bull_ids else "").split(",") if item.strip() and item.strip() != "none"]
            bear_supported = [item.strip() for item in (bear_ids.group(1) if bear_ids else "").split(",") if item.strip() and item.strip() != "none"]
            bullish_markers = (
                "trend=bullish",
                "bullish technical setup",
                "constructive technical setup",
                "bullish characteristics",
            )
            bearish_markers = (
                "trend=bearish",
                "trend is negative",
                "prevailing trend remains negative",
            )
            if any(marker in text for marker in bullish_markers) and "Bull Analyst:" in text and len(bull_supported) >= 2:
                rec = PortfolioRating.BUY
                rationale = "The bull case carried more weight because the market setup and momentum evidence outweighed the cautionary signals."
                action = "Proceed with the trade while keeping position size disciplined and risk controls explicit."
                evidence = bull_supported[:3]
            elif any(marker in text for marker in bearish_markers) and "Bear Analyst:" in text and len(bear_supported) >= 2:
                rec = PortfolioRating.SELL
                rationale = "The bear case was more persuasive because negative trend evidence outweighed the rebound narrative."
                action = "Avoid adding exposure and reduce or exit the position where appropriate."
                evidence = bear_supported[:3]
            else:
                rec = PortfolioRating.HOLD
                rationale = "The evidence does not establish a sufficient edge in either direction."
                action = "Wait for a cleaner setup before committing capital."
                evidence = []
            return ResearchPlan(
                recommendation=rec,
                rationale=rationale,
                strategic_actions=action,
                supporting_evidence_ids=evidence,
            )

        if self.schema is TraderProposal:
            if "**Recommendation**: Buy" in text:
                action = TraderAction.BUY
                sizing = "41% requested before risk manager cap"
            elif "**Recommendation**: Sell" in text:
                action = TraderAction.SELL
                sizing = "35% of current position before risk manager cap"
            else:
                action = TraderAction.HOLD
                sizing = "0% of portfolio"
            return TraderProposal(
                action=action,
                reasoning="This recommendation follows the research plan and should still be filtered through the risk-management review.",
                entry_price=None,
                stop_loss=None,
                position_sizing=sizing,
            )

        if self.schema is PortfolioDecision:
            bull_ids = re.search(r"Bull validated evidence IDs:\s*(.*)", text)
            bear_ids = re.search(r"Bear validated evidence IDs:\s*(.*)", text)
            bull_supported = [item.strip() for item in (bull_ids.group(1) if bull_ids else "").split(",") if item.strip() and item.strip() != "none"]
            bear_supported = [item.strip() for item in (bear_ids.group(1) if bear_ids else "").split(",") if item.strip() and item.strip() != "none"]
            if "**Action**: Buy" in text or "FINAL TRANSACTION PROPOSAL: **BUY**" in text:
                if len(bull_supported) >= 2:
                    rating = PortfolioRating.BUY
                    summary = "Approve the long while keeping explicit sizing discipline and downside controls in place."
                    evidence = bull_supported[:3]
                else:
                    rating = PortfolioRating.HOLD
                    summary = "Hold because the long-side evidence did not stay sufficiently grounded after validation."
                    evidence = []
            elif "**Action**: Sell" in text or "FINAL TRANSACTION PROPOSAL: **SELL**" in text:
                if len(bear_supported) >= 2:
                    rating = PortfolioRating.SELL
                    summary = "Approve the sell-side outcome because the risk debate did not justify keeping the trade on."
                    evidence = bear_supported[:3]
                else:
                    rating = PortfolioRating.HOLD
                    summary = "Hold because the sell-side evidence did not stay sufficiently grounded after validation."
                    evidence = []
            else:
                rating = PortfolioRating.HOLD
                summary = "Hold because the case is still too mixed to justify a decisive portfolio move."
                evidence = []
            return PortfolioDecision(
                rating=rating,
                executive_summary=summary,
                investment_thesis="This conclusion follows from the trader proposal and the balance of evidence presented in the risk debate.",
                price_target=None,
                time_horizon="backtest evaluation window",
                supporting_evidence_ids=evidence,
            )

        return self.llm.invoke(prompt)
