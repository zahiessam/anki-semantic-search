"""Embedding lookup, semantic search, metadata, and context boosting helpers."""

# ============================================================================
# Imports
# ============================================================================

import datetime
import hashlib

from aqt import mw
from aqt.qt import QApplication

from ..core.engine import get_embedding_for_query, load_embedding
from ..utils import load_config, log_debug


# ============================================================================
# Embedding And Context Boost Helpers
# ============================================================================

def _get_note_embedding(self, note_content, note_id=None):



    """Get embedding for a note (checks persistent storage first, then cache).







    IMPORTANT:



    - Embeddings are keyed by a hash of the *generation-time* note content,



      not necessarily the string we display in the search UI.



    - Originally we hashed the UI content, which differed (HTML stripped,



      different separators), so lookups always missed and embeddings were



      treated as "not available".







    This helper now reconstructs the same content that was used during the



    Create/Update Embeddings workflow so the hashes line up.



    """



    import hashlib







    # Lazily initialise in\xe2\u20ac\u2018memory cache



    if not hasattr(self, "_embedding_cache"):



        self._embedding_cache = {}







    # Try fast path: hash of the UI content (in case future versions align)



    ui_hash = hashlib.md5(note_content.encode()).hexdigest()



    if ui_hash in self._embedding_cache:



        return self._embedding_cache.get(ui_hash)







    # If we don't have a note id we can't reconstruct the generation



    # content reliably, so bail out early.



    if note_id is None:



        return None







    # Reconstruct the content string the embedding generator used so that



    # we derive the exact same MD5 and key.



    try:



        from aqt import mw  # Local import to avoid issues at import time



        note = mw.col.get_note(note_id)



        m = note.note_type()



        model_name = m["name"]



        flds = m["flds"]







        config = load_config()



        ntf = config.get("note_type_filter", {})



        enabled = set(ntf.get("enabled_note_types") or [])



        search_all = ntf.get("search_all_fields", False)



        ntf_fields = ntf.get("note_type_fields", {})



        use_first = ntf.get("use_first_field_fallback", True)







        # If embeddings were originally restricted to certain note types,



        # mirror that logic here. If the model wasn't enabled at generation



        # time we almost certainly won't have an embedding anyway.



        if enabled and model_name not in enabled:



            return None







        if search_all:



            indices = [



                i for i in range(min(len(note.fields), len(flds)))



                if note.fields[i].strip()



            ]



        else:



            wanted = set(



                f.lower() for f in (ntf_fields.get(model_name) or [])



            )



            if not wanted and use_first and flds:



                wanted = {flds[0]["name"].lower()}



            indices = [



                i for i, f in enumerate(flds)



                if i < len(note.fields)



                and f["name"].lower() in wanted



                and note.fields[i].strip()



            ]



        if not indices:



            return None



        content_parts = [note.fields[i] for i in indices]







        # This mirrors the generator: join with spaces and *do not* strip HTML.



        generation_content = " ".join(content_parts)



        content_hash = hashlib.md5(generation_content.encode()).hexdigest()



    except Exception:



        # Fall back to the UI hash; if that doesn't exist in storage,



        # load_embedding will just return None.



        content_hash = ui_hash







    # Check cache again with the (possibly different) generation hash



    if content_hash in self._embedding_cache:



        return self._embedding_cache.get(content_hash)







    # Then check persistent storage using the reconstructed hash



    persistent_embedding = load_embedding(note_id, content_hash)



    if persistent_embedding is not None:



        # Cache using the generation hash; we also alias under the UI hash



        # so repeated lookups for the same text are fast.



        self._embedding_cache[content_hash] = persistent_embedding



        self._embedding_cache[ui_hash] = persistent_embedding



        return persistent_embedding







    # If not found, we skip generating on-the-fly to avoid extra API usage.



    return None



def _embedding_search(self, query, notes):



    """Semantic search using precomputed embeddings (Voyage).







    Uses the configured embedding engine (Voyage or Ollama) for the query



    embedding, and expects note embeddings to have been generated ahead of



    time via the Create/Update Embeddings workflow.



    """



    try:



        import numpy as np



    except ImportError:



        log_debug("numpy not available, embedding search disabled")



        return None







    try:



        # Get query embedding via configured engine (Voyage or Ollama)



        embedding_list = get_embedding_for_query(query)



        if not embedding_list:



            log_debug("Empty query embedding from embedding engine")



            return None



        query_embedding = np.array(embedding_list)







        # Compute similarities



        scored_notes = []







        # For very large collections, limit how many notes we run full



        # embedding search over to keep things responsive.



        max_notes_for_embedding = 5000



        total_notes = len(notes)



        if total_notes > max_notes_for_embedding:



            log_debug(f"Embedding search: limiting to first {max_notes_for_embedding} of {total_notes} notes for performance.")



            notes_iter = notes[:max_notes_for_embedding]



        else:



            notes_iter = notes







        # Keep UI responsive during long embedding loops



        try:



            from aqt.qt import QApplication



        except Exception:



            QApplication = None







        for idx, note in enumerate(notes_iter):



            if QApplication is not None and idx % 500 == 0:



                QApplication.processEvents()



                log_debug(f"Embedding search progress: processed {idx}/{len(notes_iter)} notes")







            note_embedding = self._get_note_embedding(note['content'], note.get('id'))



            if note_embedding is not None:



                # Cosine similarity



                similarity = np.dot(query_embedding, note_embedding) / (



                    np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)



                )



                # Convert to 0-100 score



                score = (similarity + 1) * 50  # Normalize from [-1,1] to [0,100]



                scored_notes.append((score, note))







        scored_notes.sort(reverse=True, key=lambda x: x[0])



        return scored_notes[:80]



    except Exception as e:



        log_debug(f"Error in embedding search: {e}")



        return None



def _get_note_metadata(self, note_id):



    """Get metadata for context-aware ranking"""



    try:



        note = mw.col.get_note(note_id)



        card = note.cards()[0] if note.cards() else None







        metadata = {



            'deck_id': card.did if card else None,



            'note_type': note.note_type()['name'] if note else None,



            'mod_time': note.mod if hasattr(note, 'mod') else 0,



            'review_count': card.reps if card else 0,



            'last_review': card.rep if card else 0,



        }



        return metadata



    except:



        return {}



def _context_aware_boost(self, note, base_score, selected_note_ids=None):



    """Apply context-aware ranking boosts"""



    if not hasattr(self, 'selected_note_ids') or not self.selected_note_ids:



        selected_note_ids = set()



    else:



        selected_note_ids = self.selected_note_ids







    boost = 1.0



    metadata = self._get_note_metadata(note['id'])







    # Boost notes from same deck/note type as previously selected



    if selected_note_ids:



        try:



            selected_metadata = [self._get_note_metadata(nid) for nid in selected_note_ids]



            selected_decks = {m.get('deck_id') for m in selected_metadata if m.get('deck_id')}



            selected_types = {m.get('note_type') for m in selected_metadata if m.get('note_type')}







            if metadata.get('deck_id') in selected_decks:



                boost *= 1.2



            if metadata.get('note_type') in selected_types:



                boost *= 1.15



        except:



            pass







    # Boost recent notes (last 30 days)



    if metadata.get('mod_time'):



        try:



            from datetime import datetime, timedelta



            mod_date = datetime.fromtimestamp(metadata['mod_time'] / 1000)



            days_ago = (datetime.now() - mod_date).days



            if days_ago < 30:



                boost *= 1.1



        except:



            pass







    # Slight boost for well-reviewed notes



    if metadata.get('review_count', 0) > 10:



        boost *= 1.05







    return base_score * boost
