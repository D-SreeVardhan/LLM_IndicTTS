import React from 'react'
import type { LangCode } from '../api'

interface Props {
  lang: LangCode
  onSelect: (q: string) => void
}

const SUGGESTIONS: Record<LangCode, { category: string; questions: string[] }[]> = {
  en: [
    {
      category: 'Fees',
      questions: [
        'What is the B.Tech fee for AY 2026-27?',
        'What are the MBA programme fees?',
        'How does the fee refund policy work?',
      ],
    },
    {
      category: 'Exams',
      questions: [
        'When are the Semester End Examinations in December?',
        'What are the passing criteria for CIE and SEE?',
        'When is the UG induction for AY 2026-27?',
      ],
    },
    {
      category: 'Admissions',
      questions: [
        'What are the admission steps for RVU?',
        'What scholarships are available?',
        'What is the attendance requirement?',
      ],
    },
    {
      category: 'Placements',
      questions: [
        'Explain the One Offer Rule at RVU.',
        'What is the maximum number of placement drives I can attend?',
        'What is the Dream Offer concept?',
      ],
    },
  ],
  hi: [
    { category: 'फीस', questions: ['B.Tech की फीस क्या है?', 'MBA प्रोग्राम की फीस कितनी है?', 'फीस रिफंड नीति क्या है?'] },
    { category: 'परीक्षा', questions: ['SEE परीक्षाएँ कब होती हैं?', 'CIE में पास होने के लिए कितने अंक चाहिए?', 'हॉल टिकट कब मिलेगा?'] },
    { category: 'प्रवेश', questions: ['आरवीयू में प्रवेश कैसे होता है?', 'छात्रवृत्ति क्या उपलब्ध है?', 'उपस्थिति कितनी होनी चाहिए?'] },
    { category: 'प्लेसमेंट', questions: ['वन ऑफर रूल क्या है?', 'अधिकतम कितने प्लेसमेंट ड्राइव में भाग ले सकते हैं?', 'ड्रीम ऑफर क्या होता है?'] },
  ],
  kn: [
    { category: 'ಶುಲ್ಕ', questions: ['B.Tech ಶುಲ್ಕ ಎಷ್ಟು?', 'MBA ಕಾರ್ಯಕ್ರಮದ ಶುಲ್ಕ ಎಷ್ಟು?', 'ಶುಲ್ಕ ಮರುಪಾವತಿ ನೀತಿ ಏನು?'] },
    { category: 'ಪರೀಕ್ಷೆ', questions: ['SEE ಪರೀಕ್ಷೆಗಳು ಯಾವಾಗ?', 'ಉತ್ತೀರ್ಣರಾಗಲು ಎಷ್ಟು ಅಂಕ ಬೇಕು?', 'ಹಾಲ್ ಟಿಕೆಟ್ ಯಾವಾಗ ಸಿಗುತ್ತದೆ?'] },
    { category: 'ಪ್ರವೇಶ', questions: ['ಪ್ರವೇಶ ಪ್ರಕ್ರಿಯೆ ಹೇಗೆ?', 'ವಿದ್ಯಾರ್ಥಿವೇತನ ಏನಿದೆ?', 'ಹಾಜರಾತಿ ಎಷ್ಟಿರಬೇಕು?'] },
    { category: 'ನೇಮಕಾತಿ', questions: ['ಒನ್ ಆಫರ್ ರೂಲ್ ಏನು?', 'ಗರಿಷ್ಠ ಎಷ್ಟು ಡ್ರೈವ್‌ಗಳಲ್ಲಿ ಭಾಗವಹಿಸಬಹುದು?', 'ಡ್ರೀಮ್ ಆಫರ್ ಎಂದರೇನು?'] },
  ],
  ta: [
    { category: 'கட்டணம்', questions: ['B.Tech கட்டணம் என்ன?', 'MBA கட்டணம் எவ்வளவு?', 'கட்டண திருப்பிலை கொள்கை என்ன?'] },
    { category: 'தேர்வு', questions: ['SEE தேர்வுகள் எப்போது?', 'தேர்ச்சிக்கு எத்தனை மதிப்பெண் வேண்டும்?', 'ஹால் டிக்கெட் எப்போது கிடைக்கும்?'] },
    { category: 'சேர்க்கை', questions: ['சேர்க்கை எப்படி நடக்கும்?', 'என்ன உதவித்தொகைகள் உள்ளன?', 'வருகை சதவீதம் எவ்வளவு இருக்க வேண்டும்?'] },
    { category: 'வேலைவாய்ப்பு', questions: ['ஒரே ஒரு ஆஃபர் விதி என்ன?', 'அதிகபட்சம் எத்தனை டிரைவ்களில் கலந்துகொள்ளலாம்?', 'கனவு சலுகை என்றால் என்ன?'] },
  ],
}

export function SuggestedQuestions({ lang, onSelect }: Props) {
  const groups = SUGGESTIONS[lang] ?? SUGGESTIONS.en

  return (
    <div style={styles.wrapper}>
      {groups.map((g) => (
        <div key={g.category} style={styles.group}>
          <span style={styles.category}>{g.category}</span>
          {g.questions.map((q) => (
            <button key={q} style={styles.pill} onClick={() => onSelect(q)} className={`lang-${lang}`}>
              {q}
            </button>
          ))}
        </div>
      ))}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: '16px' },
  group: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px' },
  category: {
    fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase',
    color: 'var(--ink-muted)', fontWeight: 700, marginRight: '4px', whiteSpace: 'nowrap',
  },
  pill: {
    padding: '6px 14px',
    border: '1px solid var(--rule)',
    borderRadius: '20px',
    background: 'var(--paper-dark)',
    cursor: 'pointer',
    fontFamily: 'inherit',
    fontSize: '13px',
    color: 'var(--ink)',
    textAlign: 'left',
    transition: 'border-color 0.15s, color 0.15s',
  },
}
