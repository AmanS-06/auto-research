
from app.schemas.evidence import Citation, Evidence


def format_citation(evidence: Evidence, index: int) -> Citation:
    snippet = evidence.content[:300] + "..." if len(evidence.content) > 300 else evidence.content
    return Citation(
        id=f"[{index}]",
        source_url=evidence.source_url,
        title=evidence.title,
        snippet=snippet,
    )


def format_citations(evidence_list: list[Evidence]) -> list[Citation]:
    return [format_citation(ev, i + 1) for i, ev in enumerate(evidence_list)]


def embed_citations_in_text(text: str, citations: list[Citation]) -> str:
    for cite in citations:
        text = text.replace(f"[{cite.id}]", f"[{cite.id}]({cite.source_url})")
    return text


def generate_bibliography(citations: list[Citation]) -> str:
    if not citations:
        return ""

    lines = ["## References", ""]
    for cite in citations:
        lines.append(f"{cite.id} {cite.title}. Available at: {cite.source_url}")
    return "\n".join(lines)


def deduplicate_evidence(evidence_list: list[Evidence], threshold: float = 0.85) -> list[Evidence]:
    if not evidence_list:
        return []

    unique = []
    seen_urls = set()

    for ev in evidence_list:
        url_str = str(ev.source_url)
        if url_str in seen_urls:
            continue

        is_duplicate = False
        for existing in unique:
            if _similarity(ev.content, existing.content) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(ev)
            seen_urls.add(url_str)

    return unique


def _similarity(text1: str, text2: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, text1, text2).ratio()


def score_source_quality(evidence: Evidence) -> float:
    score = evidence.relevance_score

    domain = (
        str(evidence.source_url).split("/")[2]
        if len(str(evidence.source_url).split("/")) > 2
        else ""
    )
    trusted_domains = {
        "wikipedia.org": 0.9,
        "arxiv.org": 0.95,
        "pubmed.ncbi.nlm.nih.gov": 0.95,
        "scholar.google.com": 0.9,
        "github.com": 0.8,
        "stackoverflow.com": 0.7,
        "medium.com": 0.5,
    }

    domain_bonus = trusted_domains.get(domain, 0.5)
    score = (score + domain_bonus) / 2

    return min(max(score, 0.0), 1.0)
