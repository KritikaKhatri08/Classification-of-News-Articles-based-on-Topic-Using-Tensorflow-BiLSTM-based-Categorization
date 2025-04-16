import { create } from 'zustand';
import { supabase } from '../lib/supabase';
import { User } from '@supabase/supabase-js';

interface AuthState {
  user: User | null;
  setUser: (user: User | null) => void;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  refreshSession: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  setUser: (user) => set({ user }),
  signIn: async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      if (error.message === 'Invalid login credentials') {
        throw new Error('Invalid email or password');
      }
      throw error;
    }
    set({ user: data.user });
  },
  signUp: async (email, password) => {
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (error) {
      if (error.message.includes('User already registered')) {
        throw new Error('Email already in use');
      }
      throw error;
    }
    set({ user: data.user });
  },
  signOut: async () => {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
    set({ user: null });
  },
  refreshSession: async () => {
    try {
      const { data: { session }, error } = await supabase.auth.getSession();
      if (error) throw error;
      if (!session) {
        set({ user: null });
        return;
      }
      set({ user: session.user });
    } catch (error) {
      console.error('Failed to refresh session:', error);
      set({ user: null });
    }
  }
}));

// Initialize auth state
supabase.auth.getSession().then(({ data: { session } }) => {
  useAuthStore.setState({ user: session?.user ?? null });
});

// Listen for auth changes
supabase.auth.onAuthStateChange(async (event, session) => {
  if (event === 'SIGNED_OUT' || event === 'USER_DELETED') {
    useAuthStore.setState({ user: null });
  } else if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
    useAuthStore.setState({ user: session?.user ?? null });
  }
});

// Handle token refresh
supabase.auth.onAuthStateChange(async (event) => {
  if (event === 'TOKEN_REFRESHED') {
    await useAuthStore.getState().refreshSession();
  }
});