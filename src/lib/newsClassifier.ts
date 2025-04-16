import * as tf from '@tensorflow/tfjs';

// Enhanced category keywords with more comprehensive coverage
const categoryKeywords = {
  Technology: {
    primary: ['ai', 'technology', 'software', 'digital', 'cyber', 'tech', 'computer', 'internet', 'blockchain', 'robot', 'code'],
    secondary: ['innovation', 'startup', 'device', 'app', 'mobile', 'data', 'cloud', 'programming', 'algorithm', 'developer'],
    companies: ['google', 'apple', 'microsoft', 'amazon', 'meta', 'tesla', 'nvidia', 'intel', 'ibm', 'oracle'],
    concepts: ['artificial intelligence', 'machine learning', 'virtual reality', 'augmented reality', 'cryptocurrency', '5g', 'quantum computing', 'cybersecurity'],
    products: ['iphone', 'android', 'windows', 'linux', 'ios', 'web3', 'neural network']
  },
  Business: {
    primary: ['business', 'market', 'economy', 'finance', 'trade', 'investment', 'stock', 'revenue', 'profit'],
    secondary: ['startup', 'company', 'industry', 'corporate', 'enterprise', 'merger', 'acquisition', 'venture'],
    financial: ['nasdaq', 'dow jones', 'sp500', 'wall street', 'ipo', 'earnings', 'shares', 'dividend', 'portfolio'],
    concepts: ['quarterly report', 'market analysis', 'economic growth', 'fiscal policy', 'monetary policy', 'inflation', 'recession'],
    sectors: ['banking', 'retail', 'manufacturing', 'real estate', 'energy', 'automotive']
  },
  Politics: {
    primary: ['politics', 'government', 'election', 'policy', 'congress', 'senate', 'democrat', 'republican', 'parliament'],
    secondary: ['legislation', 'vote', 'campaign', 'political', 'president', 'administration', 'diplomatic', 'governor'],
    international: ['foreign policy', 'international relations', 'diplomacy', 'treaty', 'sanctions', 'united nations', 'eu'],
    concepts: ['democracy', 'constitution', 'bipartisan', 'legislative', 'judiciary', 'executive order'],
    events: ['summit', 'referendum', 'inauguration', 'impeachment', 'coalition']
  },
  Sports: {
    primary: ['sports', 'game', 'team', 'player', 'championship', 'tournament', 'match', 'score', 'athlete'],
    secondary: ['win', 'lose', 'victory', 'defeat', 'season', 'league', 'coach', 'stadium', 'record'],
    leagues: ['nfl', 'nba', 'mlb', 'nhl', 'fifa', 'uefa', 'olympics', 'premier league', 'formula 1'],
    concepts: ['world cup', 'super bowl', 'playoffs', 'final four', 'grand slam', 'medal', 'draft'],
    roles: ['quarterback', 'striker', 'pitcher', 'defender', 'manager', 'referee']
  },
  Entertainment: {
    primary: ['entertainment', 'movie', 'film', 'music', 'celebrity', 'actor', 'actress', 'star', 'director'],
    secondary: ['hollywood', 'tv', 'show', 'series', 'album', 'concert', 'award', 'performance', 'cast'],
    events: ['oscar', 'grammy', 'emmy', 'golden globe', 'festival', 'premiere', 'red carpet'],
    concepts: ['box office', 'streaming', 'rating', 'review', 'debut', 'sequel', 'franchise'],
    platforms: ['netflix', 'disney', 'hbo', 'spotify', 'amazon prime', 'hulu']
  },
  Health: {
    primary: ['health', 'medical', 'disease', 'treatment', 'patient', 'doctor', 'hospital', 'medicine', 'clinic'],
    secondary: ['research', 'study', 'clinical', 'therapy', 'vaccine', 'drug', 'pharmaceutical', 'diagnosis'],
    conditions: ['cancer', 'diabetes', 'heart disease', 'obesity', 'mental health', 'alzheimer', 'covid'],
    concepts: ['public health', 'healthcare', 'medical research', 'clinical trial', 'prevention', 'wellness'],
    specialists: ['surgeon', 'physician', 'nurse', 'pediatrician', 'psychiatrist']
  },
  Science: {
    primary: ['science', 'research', 'study', 'discovery', 'scientist', 'experiment', 'theory', 'evidence', 'laboratory'],
    secondary: ['scientific', 'physics', 'chemistry', 'biology', 'astronomy', 'climate', 'evolution', 'genome'],
    fields: ['quantum', 'molecular', 'genetic', 'environmental', 'neuroscience', 'biochemistry', 'astrophysics'],
    concepts: ['peer review', 'scientific method', 'breakthrough', 'hypothesis', 'observation', 'data analysis'],
    institutions: ['nasa', 'cern', 'university', 'laboratory', 'institute']
  }
};

// Create enhanced vocabulary with word importance weights
const createWeightedVocabulary = () => {
  const weightedVocab = new Map<string, number>();
  
  Object.values(categoryKeywords).forEach(category => {
    // Primary keywords get highest weight
    category.primary.forEach(keyword => {
      keyword.split(' ').forEach(word => {
        weightedVocab.set(word.toLowerCase(), 3.0);
      });
    });
    
    // Secondary keywords get medium weight
    category.secondary.forEach(keyword => {
      keyword.split(' ').forEach(word => {
        weightedVocab.set(word.toLowerCase(), 2.0);
      });
    });
    
    // Other category-specific terms get standard weight
    Object.entries(category).forEach(([key, terms]) => {
      if (!['primary', 'secondary'].includes(key)) {
        terms.forEach(term => {
          term.split(' ').forEach(word => {
            if (!weightedVocab.has(word.toLowerCase())) {
              weightedVocab.set(word.toLowerCase(), 1.5);
            }
          });
        });
      }
    });
  });
  
  return weightedVocab;
};

const weightedVocabulary = createWeightedVocabulary();

// Enhanced text preprocessing
const preprocessText = (text: string): string[] => {
  return text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 1);
};

// Calculate enhanced TF-IDF with word importance weights
const calculateWeightedTfIdf = (text: string): Map<string, number> => {
  const words = preprocessText(text);
  const wordFreq = new Map<string, number>();
  const scores = new Map<string, number>();
  
  // Calculate term frequency
  words.forEach(word => {
    wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
  });
  
  // Calculate weighted TF-IDF scores
  wordFreq.forEach((freq, word) => {
    const wordWeight = weightedVocabulary.get(word) || 1.0;
    const tf = freq / words.length;
    const idf = Math.log(1 + (weightedVocabulary.size / (1 + (freq > 0 ? 1 : 0))));
    scores.set(word, tf * idf * wordWeight);
  });
  
  return scores;
};

// Calculate semantic similarity between text and category
const calculateSemanticSimilarity = (textScores: Map<string, number>, category: string): number => {
  let similarity = 0;
  let totalWeight = 0;
  
  Object.values(categoryKeywords[category as keyof typeof categoryKeywords]).forEach(terms => {
    terms.forEach(term => {
      const words = preprocessText(term);
      words.forEach(word => {
        const wordScore = textScores.get(word) || 0;
        const wordWeight = weightedVocabulary.get(word) || 1.0;
        similarity += wordScore * wordWeight;
        totalWeight += wordWeight;
      });
    });
  });
  
  return totalWeight > 0 ? similarity / totalWeight : 0;
};

// Analyze context patterns with enhanced scoring
const analyzeContextPatterns = (text: string): Map<string, number> => {
  const scores = new Map<string, number>();
  const words = preprocessText(text);
  const textLength = words.length;
  
  Object.entries(categoryKeywords).forEach(([category, keywords]) => {
    let score = 0;
    let matches = 0;
    
    // Check for exact phrase matches
    Object.entries(keywords).forEach(([type, terms]) => {
      terms.forEach(term => {
        if (text.toLowerCase().includes(term.toLowerCase())) {
          const weight = type === 'primary' ? 4 : 
                        type === 'secondary' ? 3 : 2;
          score += weight;
          matches++;
        }
      });
    });
    
    // Calculate density-based score
    const density = matches / textLength;
    const finalScore = (score * (1 + density)) / (1 + Math.log(1 + textLength));
    scores.set(category, finalScore);
  });
  
  return scores;
};

// Main classification function with improved accuracy
export const classifyNews = (text: string): {
  category: string;
  confidence: number;
  predictions: Array<{ category: string; confidence: number }>;
} => {
  const textScores = calculateWeightedTfIdf(text);
  const contextScores = analyzeContextPatterns(text);
  const predictions = new Map<string, number>();
  
  Object.keys(categoryKeywords).forEach(category => {
    const semanticScore = calculateSemanticSimilarity(textScores, category);
    const contextScore = contextScores.get(category) || 0;
    
    // Weighted combination of scores
    const combinedScore = (semanticScore * 0.6) + (contextScore * 0.4);
    predictions.set(category, combinedScore);
  });
  
  // Normalize scores to percentages
  const maxScore = Math.max(...predictions.values());
  const normalizedPredictions = Array.from(predictions.entries())
    .map(([category, score]) => ({
      category,
      confidence: Math.min((score / maxScore) * 100, 100)
    }))
    .sort((a, b) => b.confidence - a.confidence);
  
  return {
    category: normalizedPredictions[0].category,
    confidence: normalizedPredictions[0].confidence,
    predictions: normalizedPredictions
  };
};