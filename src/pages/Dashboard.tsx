import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { History, ChevronLeft, BookOpen, LogOut } from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import { supabase } from '../lib/supabase';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { fetchNewsArticles } from '../lib/newsApi';

interface ArticleHistory {
  id: string;
  title: string;
  category: string;
  image_url: string;
  description: string;
  read_at: string;
}

interface RecommendedArticle {
  title: string;
  description: string;
  urlToImage: string;
  category: string;
  url: string;
  source: string;
  publishedAt: string;
}

const COLORS = ['#4f46e5', '#7c3aed', '#2563eb', '#7c3aed', '#3b82f6'];
const DEFAULT_CATEGORIES = ['Technology', 'Business', 'Science'];

const Dashboard = () => {
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const signOut = useAuthStore((state) => state.signOut);
  const [history, setHistory] = useState<ArticleHistory[]>([]);
  const [categoryStats, setCategoryStats] = useState<{ name: string; value: number }[]>([]);
  const [recommendations, setRecommendations] = useState<RecommendedArticle[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      setError('Failed to log out. Please try again.');
    }
  };

  const fetchRecommendations = async (categories: string[]) => {
    let allRecommendations: RecommendedArticle[] = [];
    let attempts = 0;
    const maxAttempts = 3;

    while (allRecommendations.length < 3 && attempts < maxAttempts) {
      try {
        // Try each category in sequence until we have enough recommendations
        for (const category of categories) {
          if (allRecommendations.length >= 3) break;

          const articles = await fetchNewsArticles(category);
          const newRecommendations = articles
            .filter(article => 
              // Filter out articles already in recommendations or history
              !allRecommendations.some(rec => rec.title === article.title) &&
              !history.some(hist => hist.title === article.title)
            )
            .slice(0, 3 - allRecommendations.length);

          allRecommendations = [...allRecommendations, ...newRecommendations];
        }

        // If we still don't have enough recommendations, try general news
        if (allRecommendations.length < 3) {
          const generalArticles = await fetchNewsArticles();
          const remainingCount = 3 - allRecommendations.length;
          const newRecommendations = generalArticles
            .filter(article => 
              !allRecommendations.some(rec => rec.title === article.title) &&
              !history.some(hist => hist.title === article.title)
            )
            .slice(0, remainingCount);

          allRecommendations = [...allRecommendations, ...newRecommendations];
        }
      } catch (error) {
        console.error('Error fetching recommendations:', error);
      }
      
      attempts++;
    }

    return allRecommendations;
  };

  const fetchUserHistory = async () => {
    if (!user) return;
    setIsLoading(true);
    setError(null);

    try {
      // Fetch user's reading history
      const { data: historyData, error: historyError } = await supabase
        .from('user_history')
        .select('*')
        .eq('user_id', user.id)
        .order('read_at', { ascending: false })
        .limit(10);

      if (historyError) throw historyError;

      setHistory(historyData || []);

      // Calculate category statistics
      const categories = historyData?.reduce((acc: { [key: string]: number }, article) => {
        acc[article.category] = (acc[article.category] || 0) + 1;
        return acc;
      }, {});

      const categoryStatsData = Object.entries(categories || {}).map(([name, value]) => ({
        name,
        value
      }));

      setCategoryStats(categoryStatsData);

      // Get recommendation categories
      let recommendationCategories: string[];
      if (categoryStatsData.length > 0) {
        // Use top categories from user history
        recommendationCategories = categoryStatsData
          .sort((a, b) => b.value - a.value)
          .map(cat => cat.name)
          .slice(0, 3);
      } else {
        // Use default categories if no history
        recommendationCategories = DEFAULT_CATEGORIES;
      }

      // Fetch and set recommendations
      const recommendedArticles = await fetchRecommendations(recommendationCategories);
      setRecommendations(recommendedArticles);

    } catch (err) {
      const error = err as Error;
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUserHistory();
  }, [user]);

  const handleArticleClick = async (article: RecommendedArticle) => {
    try {
      await supabase.from('user_history').insert({
        user_id: user?.id,
        article_id: article.title,
        category: article.category,
        title: article.title,
        image_url: article.urlToImage,
        description: article.description
      });
      
      window.open(article.url, '_blank');
      
      // Refresh recommendations after clicking
      fetchUserHistory();
    } catch (err) {
      const error = err as Error;
      setError(`Failed to save article history: ${error.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <header className="flex justify-between items-center mb-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
          >
            <ChevronLeft className="h-5 w-5" />
            Back to Home
          </button>
          <h1 className="text-2xl font-bold text-gray-900">Your Dashboard</h1>
        </div>
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
        >
          <LogOut className="h-5 w-5" />
          Logout
        </button>
      </header>

      {error && (
        <div className="mb-8 bg-red-50 text-red-700 p-4 rounded-lg">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center gap-2 mb-6">
            <History className="h-6 w-6 text-indigo-600" />
            <h2 className="text-xl font-semibold">Reading History</h2>
          </div>

          {history.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No reading history yet. Start reading articles to see them here!
            </div>
          ) : (
            <div className="space-y-4">
              {history.map((article) => (
                <div key={article.id} className="flex gap-4 p-4 bg-gray-50 rounded-lg">
                  <img
                    src={article.image_url}
                    alt={article.title}
                    className="w-24 h-24 object-cover rounded-lg"
                  />
                  <div>
                    <span className="inline-block px-2 py-1 text-sm font-semibold text-indigo-600 bg-indigo-50 rounded-full mb-2">
                      {article.category}
                    </span>
                    <h3 className="font-semibold text-gray-900">{article.title}</h3>
                    <p className="text-sm text-gray-600">
                      Read on {new Date(article.read_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <div className="space-y-8">
          <section className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center gap-2 mb-6">
              <BookOpen className="h-6 w-6 text-indigo-600" />
              <h2 className="text-xl font-semibold">Reading Analytics</h2>
            </div>

            {categoryStats.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No reading analytics yet. Read some articles to see your interests!
              </div>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={categoryStats}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {categoryStats.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </section>

          <section className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-6">Recommended for You</h2>
            {isLoading ? (
              <div className="flex justify-center items-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
              </div>
            ) : recommendations.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No recommendations available at the moment. Please try again later.
              </div>
            ) : (
              <div className="space-y-4">
                {recommendations.map((article, index) => (
                  <div
                    key={index}
                    onClick={() => handleArticleClick(article)}
                    className="flex gap-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer"
                  >
                    <img
                      src={article.urlToImage}
                      alt={article.title}
                      className="w-24 h-24 object-cover rounded-lg"
                      onError={(e) => {
                        e.currentTarget.src = 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=800';
                      }}
                    />
                    <div>
                      <span className="inline-block px-2 py-1 text-sm font-semibold text-indigo-600 bg-indigo-50 rounded-full mb-2">
                        {article.category}
                      </span>
                      <h3 className="font-semibold text-gray-900">{article.title}</h3>
                      <p className="text-sm text-gray-600 line-clamp-2">{article.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;