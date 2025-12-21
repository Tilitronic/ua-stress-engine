#!/usr/bin/env python3
"""
Demo: Ukrainian NLP Pipeline with Complex Text

Tests the pipeline with heteronyms and complex morphology.
"""

from src.nlp.pipeline import UkrainianPipeline


def main():
    """Run pipeline demo with test text."""
    
    text = """Замок стоїть великий на горі, 
старі замки іржавіють на дворі,
гортає атлас принц, а його кінь 
жує піжам атлас, тканини синь
зникає в роті велетня-тарпана. 

Коли немає жодної блохи собаці живеться добре, 
бо ці надокучливі блохи справді втомили собаку."""
    
    print("=" * 80)
    print("UKRAINIAN NLP PIPELINE DEMO")
    print("=" * 80)
    print(f"\nInput text:\n{text}\n")
    print("=" * 80)
    
    # Process with pipeline
    print("\n🚀 Initializing pipeline...")
    with UkrainianPipeline() as pipeline:
        print("✓ Pipeline ready\n")
        
        print("🔄 Processing text through pipeline:")
        print("   Stage 1: Tokenization (spaCy)")
        print("   Stage 2: Stress resolution (morphology matching)")
        print("   Stage 3: Phonetic transcription (IPA)\n")
        
        result = pipeline.process(text)
        
        print("=" * 80)
        print("PIPELINE RESULTS")
        print("=" * 80)
        
        # Overall statistics
        print(f"\n📊 Statistics:")
        print(f"   Total tokens: {result.total_tokens}")
        print(f"   Words processed: {result.words_processed}")
        print(f"   Words with stress: {result.words_with_stress}")
        print(f"   Words with phonetic: {result.words_with_phonetic}")
        print(f"   Stress coverage: {result.stress_coverage:.1f}%")
        print(f"   Phonetic coverage: {result.phonetic_coverage:.1f}%")
        
        # Process each sentence
        print(f"\n📝 Detailed Analysis:\n")
        
        for sent_idx, sentence in enumerate(result.sentences, 1):
            print(f"\n{'─' * 80}")
            print(f"Sentence {sent_idx}: {sentence.text.strip()}")
            print(f"{'─' * 80}")
            
            # Find interesting tokens (heteronyms, stressed words)
            for token in sentence.tokens:
                if not token.is_alpha or token.is_punct:
                    continue
                
                # Format output
                word_display = f"{token.text:15}"
                pos_display = f"{token.pos:6}"
                
                # Stress info
                if token.stress_pattern:
                    stress_display = f"{token.stress_pattern:15}"
                    confidence = f"({token.stress_confidence}, {token.stress_match_score:.2f})"
                else:
                    stress_display = f"{'—':15}"
                    confidence = "(none)"
                
                # Phonetic
                if token.phonetic:
                    phonetic_display = f"[{token.phonetic}]"
                else:
                    phonetic_display = "—"
                
                # Morphology summary
                if token.morph:
                    morph_items = [f"{k}={v}" for k, v in list(token.morph.items())[:2]]
                    morph_display = ", ".join(morph_items)
                    if len(token.morph) > 2:
                        morph_display += ", ..."
                else:
                    morph_display = "—"
                
                print(f"  {word_display} {pos_display} │ {stress_display} {confidence:20} │ {phonetic_display:20} │ {morph_display}")
        
        # Highlight heteronyms
        print(f"\n{'=' * 80}")
        print("🎯 HETERONYM ANALYSIS (Words with Multiple Stress Patterns)")
        print(f"{'=' * 80}\n")
        
        heteronyms = {}
        for sentence in result.sentences:
            for token in sentence.tokens:
                if token.is_alpha and not token.is_punct:
                    word_lower = token.text.lower()
                    if word_lower not in heteronyms:
                        heteronyms[word_lower] = []
                    heteronyms[word_lower].append({
                        'original': token.text,
                        'stress': token.stress_pattern,
                        'stress_pos': token.stress_position,
                        'pos': token.pos,
                        'morph': token.morph,
                        'phonetic': token.phonetic,
                        'confidence': token.stress_confidence,
                        'score': token.stress_match_score,
                    })
        
        # Show words that appear multiple times
        for word, occurrences in heteronyms.items():
            if len(occurrences) > 1 or word in ['замок', 'замки', 'атлас']:
                print(f"\n📌 Word: '{word}' ({len(occurrences)} occurrence(s))")
                for idx, occ in enumerate(occurrences, 1):
                    print(f"\n   Occurrence {idx}:")
                    print(f"      Text: {occ['original']}")
                    print(f"      POS: {occ['pos']}")
                    print(f"      Morphology: {occ['morph']}")
                    print(f"      Stress: {occ['stress']} (position: {occ['stress_pos']})")
                    print(f"      Confidence: {occ['confidence']} (score: {occ['score']:.2f})")
                    print(f"      Phonetic: [{occ['phonetic']}]")
        
        print(f"\n{'=' * 80}")
        print("✨ Pipeline processing complete!")
        print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
