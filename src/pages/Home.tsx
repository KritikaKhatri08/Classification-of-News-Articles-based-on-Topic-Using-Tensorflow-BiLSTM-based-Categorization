import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Newspaper, Search, Brain, History, Sun, Moon, SplitSquareHorizontal, LayoutGridIcon } from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import { supabase } from '../lib/supabase';
import { fetchNewsArticles } from '../lib/newsApi';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, RadialBarChart, RadialBar } from 'recharts';

const NEWS_CATEGORIES = ['Business', 'Technology', 'Politics', 'Sports', 'Entertainment'];

interface Article {
  title: string;
  description: string;
  urlToImage: string;
  category: string;
  url: string;
  source: string;
  publishedAt: string;
}

interface ClassificationResult {
  category: string;
  confidence: number;
  probabilities: Record<string, number>;
}

const COLORS = {
  Technology: '#4f46e5',
  Business: '#06b6d4',
  Politics: '#ec4899',
  Sports: '#22c55e',
  Entertainment: '#f59e0b',
  Health: '#ef4444',
  Science: '#8b5cf6',
  tech: '#4f46e5',
  business: '#06b6d4',
  politics: '#ec4899',
  sport: '#22c55e',
  entertainment: '#f59e0b',
  health: '#ef4444',
  science: '#8b5cf6'
};

const Home = () => {
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const [articles, setArticles] = useState<Article[]>([]);
  const [searchCategory, setSearchCategory] = useState('');
  const [classificationText, setClassificationText] = useState('');
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'bar' | 'radial'>('bar');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showClassification, setShowClassification] = useState(true);

  useEffect(() => {
    fetchNews();
    // Check for user's preferred theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    }
    
    // Check for user's preferred layout
    const savedLayout = localStorage.getItem('showClassification');
    if (savedLayout === 'false') {
      setShowClassification(false);
    }
  }, []);

  useEffect(() => {
    // Update the document's class list based on dark mode state
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);
  
  useEffect(() => {
    // Save user's layout preference
    localStorage.setItem('showClassification', showClassification.toString());
  }, [showClassification]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };
  
  const toggleClassificationPanel = () => {
    setShowClassification(!showClassification);
  };

  const fetchNews = async (category?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const newsArticles = await fetchNewsArticles(category);
      setArticles(newsArticles);
    } catch (err) {
      const error = err as Error;
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleArticleClick = async (article: Article) => {
    try {
      const { error: supabaseError } = await supabase.from('user_history').insert({
        user_id: user?.id,
        article_id: article.title,
        category: article.category,
        title: article.title,
        image_url: article.urlToImage,
        description: article.description
      });

      if (supabaseError) throw supabaseError;
      
      window.open(article.url, '_blank');
    } catch (err) {
      const error = err as Error;
      setError(`Failed to save article history: ${error.message}`);
    }
  };
  
  const classifyNewsText = async () => {
    if (!classificationText.trim() || isClassifying) return;
  
    setIsClassifying(true);
    setError(null);
  
    try {
      const response = await fetch(
        'https://krutinraj-news-classification.hf.space/predict',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: classificationText,
          }),
        }
      );
  
      if (!response.ok) throw new Error('Failed to classify text');
  
      const result = await response.json();
      console.log(result);
  
      setClassificationResult(result);
    } catch (err) {
      const error = err;
      setError(`Classification failed: ${error.message}`);
    } finally {
      setIsClassifying(false);
    }
  };
  
  const renderConfidenceChart = () => {
    if (!classificationResult || !classificationResult.probabilities) return null;

    // Transform probabilities to array for chart
    const data = Object.entries(classificationResult.probabilities).map(([category, confidence]) => ({
      category: category.charAt(0).toUpperCase() + category.slice(1), // Capitalize first letter
      confidence: confidence * 100,
      fill: COLORS[category as keyof typeof COLORS]
    }));

    if (selectedView === 'bar') {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="category"
              angle={-45}
              textAnchor="end"
              height={80}
              interval={0}
              tick={{ fill: isDarkMode ? '#9ca3af' : '#4B5563', fontSize: 12 }}
            />
            <YAxis
              domain={[0, 100]}
              label={{
                value: 'Confidence (%)',
                angle: -90,
                position: 'insideLeft',
                style: { textAnchor: 'middle', fill: isDarkMode ? '#9ca3af' : '#4B5563' }
              }}
              tick={{ fill: isDarkMode ? '#9ca3af' : '#4B5563' }}
            />
            <Tooltip
              formatter={(value: number) => [`${value.toFixed(1)}%`, 'Confidence']}
              contentStyle={{
                backgroundColor: isDarkMode ? 'rgba(30, 41, 59, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                border: isDarkMode ? '1px solid #334155' : '1px solid #E5E7EB',
                borderRadius: '6px',
                color: isDarkMode ? '#e5e7eb' : 'inherit'
              }}
            />
            <Bar
              dataKey="confidence"
              name="Confidence"
              animationDuration={1000}
              label={{
                position: 'top',
                formatter: (value: number) => `${value.toFixed(1)}%`,
                fill: isDarkMode ? '#e5e7eb' : '#4B5563',
                fontSize: 12
              }}
            />
          </BarChart>
        </ResponsiveContainer>
      );
    }

    return (
      <ResponsiveContainer width="100%" height={400}>
        <RadialBarChart
          innerRadius="30%"
          outerRadius="100%"
          data={data}
          startAngle={180}
          endAngle={-180}
        >
          <RadialBar
            label={{
              fill: isDarkMode ? '#e5e7eb' : '#4B5563',
              position: 'insideStart',
              formatter: (value: number) => `${value.toFixed(1)}%`
            }}
            background
            dataKey="confidence"
            name="Confidence"
          />
          <Legend
            iconSize={10}
            layout="vertical"
            verticalAlign="middle"
            align="right"
            formatter={(value, entry) => {
              // @ts-ignore - entry has additional properties
              return entry.payload.category;
            }}
          />
          <Tooltip
            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Confidence']}
            labelFormatter={(index) => data[index].category}
            contentStyle={{
              backgroundColor: isDarkMode ? 'rgba(30, 41, 59, 0.95)' : 'rgba(255, 255, 255, 0.95)',
              border: isDarkMode ? '1px solid #334155' : '1px solid #E5E7EB',
              borderRadius: '6px',
              color: isDarkMode ? '#e5e7eb' : 'inherit'
            }}
          />
        </RadialBarChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className={`min-h-screen p-8 ${isDarkMode ? 'bg-slate-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      <header className="flex justify-between items-center mb-8">
        <div className="flex items-center gap-2">
          <Newspaper className={`h-8 w-8 ${isDarkMode ? 'text-indigo-400' : 'text-indigo-600'}`} />
          <h1 className="text-2xl font-bold">News Classification</h1>
        </div>
        <div className="flex gap-3">
          <button
            onClick={toggleClassificationPanel}
            className={`p-2 rounded-lg ${isDarkMode ? 'bg-slate-700 hover:bg-slate-600' : 'bg-gray-200 hover:bg-gray-300'}`}
            aria-label={showClassification ? "Hide classification panel" : "Show classification panel"}
            title={showClassification ? "Hide classification panel" : "Show classification panel"}
          >
            {showClassification ? 
              <LayoutGridIcon size={20} className={isDarkMode ? 'text-indigo-400' : 'text-indigo-700'} /> : 
              <SplitSquareHorizontal size={20} className={isDarkMode ? 'text-indigo-400' : 'text-indigo-700'} />
            }
          </button>
          <button
            onClick={toggleTheme}
            className={`p-2 rounded-lg ${isDarkMode ? 'bg-slate-700 text-yellow-300 hover:bg-slate-600' : 'bg-gray-200 text-indigo-700 hover:bg-gray-300'}`}
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button
            onClick={() => navigate('/dashboard')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
              isDarkMode ? 'bg-indigo-600 text-white hover:bg-indigo-700' : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            <History className="h-5 w-5" />
            Dashboard
          </button>
        </div>
      </header>

      <div className={`grid gap-8 ${showClassification ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
        <section className="space-y-6">
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
              <input
                type="text"
                placeholder="Search by category..."
                value={searchCategory}
                onChange={(e) => setSearchCategory(e.target.value)}
                className={`w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:border-transparent ${
                  isDarkMode 
                    ? 'bg-slate-800 border-slate-700 text-gray-100 focus:ring-indigo-500' 
                    : 'bg-white border-gray-300 text-gray-900 focus:ring-indigo-500'
                }`}
              />
            </div>
            <button
              onClick={() => fetchNews(searchCategory)}
              className={`px-4 py-2 rounded-lg ${
                isDarkMode ? 'bg-indigo-600 text-white hover:bg-indigo-700' : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              Search
            </button>
          </div>

          <div className="flex gap-2 flex-wrap">
            {NEWS_CATEGORIES.map((category) => (
              <button
                key={category}
                onClick={() => {
                  setSearchCategory(category);
                  fetchNews(category);
                }}
                className={`px-3 py-1 rounded-full ${
                  isDarkMode 
                    ? 'bg-indigo-900 text-indigo-300 hover:bg-indigo-800' 
                    : 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                }`}
              >
                {category}
              </button>
            ))}
          </div>

          {error && (
            <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-red-900 text-red-200' : 'bg-red-50 text-red-700'}`}>
              {error}
            </div>
          )}

          {isLoading ? (
            <div className="flex justify-center items-center py-12">
              <div className={`animate-spin rounded-full h-12 w-12 border-b-2 ${isDarkMode ? 'border-indigo-400' : 'border-indigo-600'}`}></div>
            </div>
          ) : (
            <div className={`grid gap-6 ${showClassification ? '' : 'md:grid-cols-2 lg:grid-cols-3'}`}>
              {articles.map((article, index) => (
                <article
                  key={index}
                  className={`rounded-lg overflow-hidden hover:shadow-lg transition-shadow cursor-pointer ${
                    isDarkMode ? 'bg-slate-800 shadow-slate-700' : 'bg-white shadow-md'
                  }`}
                  onClick={() => handleArticleClick(article)}
                >
                  <img
                    src={article.urlToImage}
                    alt={article.title}
                    className="w-full h-48 object-cover"
                    onError={(e) => {
                      e.currentTarget.src = 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=800';
                    }}
                  />
                  <div className="p-4">
                    <div className="flex justify-between items-start mb-2">
                      <span className={`inline-block px-2 py-1 text-sm font-semibold rounded-full ${
                        isDarkMode ? 'bg-indigo-900 text-indigo-300' : 'bg-indigo-50 text-indigo-600'
                      }`}>
                        {article.category}
                      </span>
                      <span className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>{article.source}</span>
                    </div>
                    <h2 className={`text-xl font-semibold mb-2 ${isDarkMode ? 'text-gray-100' : 'text-gray-900'}`}>{article.title}</h2>
                    <p className={`mb-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>{article.description}</p>
                    <div className="flex justify-between items-center">
                      <span className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {new Date(article.publishedAt).toLocaleDateString()}
                      </span>
                      <span className={`${isDarkMode ? 'text-indigo-400 hover:text-indigo-300' : 'text-indigo-600 hover:text-indigo-800'}`}>Read more →</span>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        {showClassification && (
          <section className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-slate-800' : 'bg-white'}`}>
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Brain className={`h-6 w-6 ${isDarkMode ? 'text-indigo-400' : 'text-indigo-600'}`} />
                <h2 className="text-xl font-semibold">News Classification</h2>
              </div>
              
              {classificationResult && (
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedView('bar')}
                    className={`px-3 py-1 rounded-lg text-sm ${
                      selectedView === 'bar'
                        ? isDarkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-600 text-white'
                        : isDarkMode ? 'bg-slate-700 text-gray-200 hover:bg-slate-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    Bar Chart
                  </button>
                  <button
                    onClick={() => setSelectedView('radial')}
                    className={`px-3 py-1 rounded-lg text-sm ${
                      selectedView === 'radial'
                        ? isDarkMode ? 'bg-indigo-600 text-white' : 'bg-indigo-600 text-white'
                        : isDarkMode ? 'bg-slate-700 text-gray-200 hover:bg-slate-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    Radial Chart
                  </button>
                </div>
              )}
            </div>

            <textarea
              value={classificationText}
              onChange={(e) => setClassificationText(e.target.value)}
              placeholder="Paste news article text here for classification..."
              className={`w-full h-40 p-4 border rounded-lg mb-4 focus:ring-2 focus:border-transparent ${
                isDarkMode 
                  ? 'bg-slate-700 border-slate-600 text-gray-100 focus:ring-indigo-500 placeholder-gray-400' 
                  : 'bg-white border-gray-300 text-gray-900 focus:ring-indigo-500'
              }`}
            />

            <button
              onClick={classifyNewsText}
              disabled={!classificationText || isClassifying}
              className={`w-full px-4 py-2 rounded-lg flex items-center justify-center gap-2 ${
                !classificationText || isClassifying
                  ? isDarkMode ? 'bg-gray-600 cursor-not-allowed' : 'bg-gray-400 cursor-not-allowed'
                  : isDarkMode ? 'bg-indigo-600 text-white hover:bg-indigo-700' : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              {isClassifying ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Classifying...
                </>
              ) : (
                'Classify Text'
              )}
            </button>

            {classificationResult && (
              <div className="mt-6">
                {renderConfidenceChart()}
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
};

export default Home;