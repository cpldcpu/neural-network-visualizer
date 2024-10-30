import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import Pages from "vite-plugin-pages";
import getRepoName from "git-repo-name";

export default defineConfig(({ command, mode }) => {
    const isGitHubPages = mode === 'github-pages'

  // Set the base path manually
    const base = isGitHubPages ? `/neural-network-visualizer/` : '/';

    return {
      plugins: [react(), Pages()],
      base: base,
      build: {
        outDir: isGitHubPages ? 'dist-github' : 'dist',
        sourcemap: true,
      },
      resolve: {
        alias: {
          "@": path.resolve(__dirname, "./src"),
        },
      },
    }
  })