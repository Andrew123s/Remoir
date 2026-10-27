"""Question answering, retrieval-only search and conversation management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.core.security import require_api_key
from app.dependencies import Container, get_container
from app.schemas import (
    AskRequest,
    AskResponse,
    ConversationOut,
    EvidenceAssessmentOut,
    GroundingOut,
    MessageResponse,
    PassageOut,
    SearchRequest,
    SearchResponse,
)
from app.services.rag_pipeline import passage_to_dict

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])
CONV_ID = Path(pattern=r"^[a-f0-9]{32}$")


def resolve_scope(c: Container, document_ids: list[str] | None, collection_id: str | None) -> list[str] | None:
    """Merge explicit document ids and a collection into one validated scope."""
    scope: list[str] = []
    if collection_id:
        collection = c.store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found.")
        scope.extend(collection["document_ids"])
        if not scope:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The collection contains no documents.")
    if document_ids:
        unknown = [d for d in document_ids if not c.store.get_document(d)]
        if unknown:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown document ids: {', '.join(unknown[:5])}")
        scope = [d for d in document_ids if d in scope] if collection_id else list(document_ids)
        if not scope:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "None of the selected documents are in the collection.")
    return list(dict.fromkeys(scope)) or None


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, c: Container = Depends(get_container)):
    """Retrieve passages without generating an answer (useful for evaluating retrieval)."""
    opts = req.retrieval
    scope = resolve_scope(c, req.document_ids, req.collection_id)
    result = c.retriever.retrieve(
        req.query, document_ids=scope, top_k=opts.top_k if opts else None,
        strategy=opts.strategy if opts else None, rerank=opts.rerank if opts else None,
    )
    labels = c.pipeline._page_labels({p.document_id for p in result.passages})
    return SearchResponse(
        query=result.query, strategy=result.strategy,
        results=[PassageOut(**passage_to_dict(p, f"S{i + 1}", labels.get(p.document_id))) for i, p in enumerate(result.passages)],
        timings_ms={k: round(v, 2) for k, v in result.timings.items()},
    )


@router.post("/chat/ask", response_model=AskResponse)
def ask(req: AskRequest, c: Container = Depends(get_container)):
    """Answer a question from the indexed documents with verified citations."""
    if req.conversation_id and not c.store.conversation_exists(req.conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    conversation_id = req.conversation_id or c.store.create_conversation()
    history = c.store.get_turns(conversation_id, c.settings.conversation.max_turns * 2)
    scope = resolve_scope(c, req.document_ids, req.collection_id)
    opts = req.retrieval
    result = c.pipeline.answer(
        req.question, document_ids=scope, history=history, top_k=opts.top_k if opts else None,
        strategy=opts.strategy if opts else None, rerank=opts.rerank if opts else None,
    )
    c.store.add_turn(conversation_id, "user", req.question, scope)
    c.store.add_turn(conversation_id, "assistant", result.answer, list(dict.fromkeys(s["document_id"] for s in result.sources)))

    report = result.citation_report
    grounding = None
    if report is not None:
        grounding = GroundingOut(
            status=report.grounding_status, invalid_citations=report.invalid_citations,
            claims=[c_.__dict__ for c_ in report.claims], unsupported_claims=len(report.unsupported_claims),
        )
    return AskResponse(
        conversation_id=conversation_id, answer=result.answer, status=result.status,
        citations=[PassageOut(**s) for s in result.sources], evidence=[PassageOut(**e) for e in result.evidence],
        grounding=grounding, evidence_assessment=EvidenceAssessmentOut(**result.assessment.__dict__),
        retrieval_query=result.retrieval_query, scope_document_ids=result.scope_document_ids,
        comparison=result.comparison, provider=result.provider, model=result.model,
        timings_ms={k: round(v, 2) for k, v in result.timings.items()},
    )


@router.post("/chat/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def new_conversation(c: Container = Depends(get_container)):
    return ConversationOut(conversation_id=c.store.create_conversation(), turns=[])


@router.get("/chat/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str = CONV_ID, c: Container = Depends(get_container)):
    if not c.store.conversation_exists(conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    turns = c.store.get_turns(conversation_id, 1000)
    return ConversationOut(
        conversation_id=conversation_id,
        turns=[{"role": t["role"], "content": t["content"], "created_at": t["created_at"]} for t in turns],
    )


@router.delete("/chat/conversations/{conversation_id}", response_model=MessageResponse)
def delete_conversation(conversation_id: str = CONV_ID, c: Container = Depends(get_container)):
    if not c.store.delete_conversation(conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    return MessageResponse(message="Conversation deleted.")
