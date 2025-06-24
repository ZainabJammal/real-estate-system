// import { defineConfig } from "vite";
// import react from "@vitejs/plugin-react";

// // https://vite.dev/config/
// export default defineConfig({
//   plugins: [react()],
//   server: {
//     proxy: {
//       '/api': 'http://localhost:8000',
//     },
//     port: 3000,
//     strictPort: true,
//   },
// });

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Your existing port configuration is fine
    port: 3000,
    strictPort: true,
    // --- Modified Proxy Configuration ---
    // proxy: {
    //   '/api': { // Any request path that starts with /api
    //     target: 'http://localhost:8000', // Will be forwarded to your Quart backend
    //     changeOrigin: true, // This is important for many backend setups
    //     // secure: false, // Usually not needed for http targets
    //     // rewrite: (path) => path.replace(/^\/api/, '') // Only if your backend routes DON'T start with /api
    //                                                   // Your backend route /api/transaction_filters DOES start with /api
    //                                                   // so you generally DON'T need this rewrite.
    //                                                   // If your backend route was just /transaction_filters, then you would.
    //   }
      // If you had other paths to proxy, like '/forecast_transaction' (if it didn't start with /api)
      // you could add another entry:
      // '/some_other_path': {
      //   target: 'http://localhost:8000',
      //   changeOrigin: true,
      // }


    },
  },
});