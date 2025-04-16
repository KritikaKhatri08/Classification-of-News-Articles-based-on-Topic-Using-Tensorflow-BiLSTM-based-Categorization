import axios from 'axios';

const NEWS_API_KEY = import.meta.env.VITE_NEWS_API_KEY;
const NEWS_API_BASE_URL = 'https://newsapi.org/v2';

// List of CORS proxies to try in order
const CORS_PROXIES = [
  'https://api.allorigins.win/raw?url=',
  'https://api.codetabs.com/v1/proxy?quest=',
  'https://corsproxy.io/?'
];

export interface NewsApiArticle {
  source: {
    id: string | null;
    name: string;
  };
  author: string | null;
  title: string;
  description: string;
  url: string;
  urlToImage: string;
  publishedAt: string;
  content: string;
}

export interface NewsApiResponse {
  status: string;
  totalResults: number;
  articles: NewsApiArticle[];
}

export interface NewsError extends Error {
  code?: string;
  response?: {
    status: number;
    data: any;
  };
}

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const fetchWithProxy = async (url: string, proxyUrl: string) => {
  const finalUrl = `${proxyUrl}${encodeURIComponent(url)}`;
  
  const response = await axios.get<NewsApiResponse>(finalUrl, {
    timeout: 15000,
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
      'X-Requested-With': 'XMLHttpRequest',
      'origin': window.location.origin
    }
  });

  if (!response.data || typeof response.data !== 'object') {
    throw new Error('Invalid response format');
  }

  if (response.data.status === 'error') {
    throw new Error(response.data.message || 'News API returned an error');
  }

  if (!Array.isArray(response.data.articles)) {
    throw new Error('Invalid articles data received');
  }

  return response;
};

const fetchWithRetryAndFallback = async (url: string, retries = 2) => {
  let lastError;

  // Try each proxy
  for (const proxy of CORS_PROXIES) {
    // Try multiple times with the same proxy
    for (let i = 0; i < retries; i++) {
      try {
        return await fetchWithProxy(url, proxy);
      } catch (error) {
        lastError = error;
        
        if (axios.isAxiosError(error)) {
          if (error.response?.status === 429) {
            // Rate limit - wait longer before retry
            await delay(2000 * (i + 1));
            continue;
          }
          
          if (error.response?.status === 401) {
            throw new Error('Invalid API key. Please check your configuration.');
          }

          if (error.response?.status === 426) {
            throw new Error('Please upgrade to a paid plan for this feature.');
          }
        }
        
        // Wait before trying again
        await delay(1000 * (i + 1));
      }
    }
  }

  // If we get here, all proxies failed
  throw lastError;
};

const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    if (error.response?.status === 429) {
      return 'Rate limit exceeded. Please try again later.';
    } else if (error.response?.status === 401) {
      return 'Invalid API key. Please check your configuration.';
    } else if (error.code === 'ECONNABORTED') {
      return 'Request timed out. Please try again.';
    } else if (error.response?.status === 426) {
      return 'Please upgrade to a paid plan for this feature.';
    } else if (error.code === 'ERR_NETWORK') {
      return 'Connection error. Please check your internet connection and try again.';
    } else if (error.response?.data?.message) {
      return `News API Error: ${error.response.data.message}`;
    }
    return 'Failed to fetch news. Please try again later.';
  }
  return error instanceof Error ? error.message : 'An unexpected error occurred.';
};

export const fetchNewsArticles = async (category?: string) => {
  try {
    const endpoint = category 
      ? `${NEWS_API_BASE_URL}/top-headlines?category=${category.toLowerCase()}&pageSize=100&country=us&apiKey=${NEWS_API_KEY}`
      : `${NEWS_API_BASE_URL}/top-headlines?pageSize=100&country=us&apiKey=${NEWS_API_KEY}`;

    const response = await fetchWithRetryAndFallback(endpoint);

    return response.data.articles
      .filter(article => 
        article && 
        typeof article === 'object' && 
        article.title && 
        article.description
      )
      .map(article => ({
        title: article.title.trim(),
        description: article.description.trim(),
        urlToImage: article.urlToImage || 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=800',
        category: category || 'General',
        url: article.url,
        source: article.source?.name || 'Unknown Source',
        publishedAt: article.publishedAt
      }));
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('News API Error:', {
      message: errorMessage,
      status: axios.isAxiosError(error) ? error.response?.status : undefined,
      code: axios.isAxiosError(error) ? error.code : undefined
    });
    throw new Error(errorMessage);
  }
};