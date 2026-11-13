import { defineRailway, github, preserve, project, service } from "railway/iac";

export default defineRailway(() => {
  const web = service("web", {
    source: github("haomingkoo/photo-id-studio", { checkSuites: false }),
    build: { builder: "DOCKERFILE", dockerfilePath: "Dockerfile" },
    deploy: {
      healthcheckPath: "/api/health",
      healthcheckTimeout: 120,
      restartPolicyType: "ON_FAILURE",
      restartPolicyMaxRetries: 10,
    },
    start: "python run.py",
    replicas: { "asia-southeast1-eqsg3a": 1 },
    domains: ["studio.kooexperience.com"],
    env: {
      PHOTO_API_ADMIN_API_KEY: preserve(),
      PHOTO_API_ADMIN_AUTH_BLOCK_SEC: preserve(),
      PHOTO_API_ADMIN_AUTH_HEADER: preserve(),
      PHOTO_API_ADMIN_AUTH_MAX_FAILS: preserve(),
      PHOTO_API_ADMIN_AUTH_WINDOW_SEC: preserve(),
      PHOTO_API_ADMIN_ENABLED: preserve(),
      PHOTO_API_BG_STRICT_WHITE: preserve(),
      PHOTO_API_BURST: preserve(),
      PHOTO_API_CONTRACT: preserve(),
      PHOTO_API_CONTRACT_VERSION: preserve(),
      PHOTO_API_DAILY_LIMIT: preserve(),
      PHOTO_API_ENABLE_REMBG: preserve(),
      PHOTO_API_LOG_ANALYZE_METRICS: preserve(),
      PHOTO_API_MAX_DECODE_MEGAPIXELS: preserve(),
      PHOTO_API_MAX_INFLIGHT: preserve(),
      PHOTO_API_MAX_UPLOAD_MB: preserve(),
      PHOTO_API_PUBLIC_APP_URL: preserve(),
      PHOTO_API_PUBLIC_REPO_URL: preserve(),
      PHOTO_API_RATE_PER_MIN: preserve(),
      PHOTO_API_REMBG_IDLE_UNLOAD_SEC: preserve(),
      PHOTO_API_REMBG_LAZY_LOAD: preserve(),
      PHOTO_API_SERVICE_ID: preserve(),
      PHOTO_API_SERVICE_NAME: preserve(),
      PHOTO_API_STORAGE_POLICY: preserve(),
      PHOTO_API_TRUST_XFF: preserve(),
    },
  });

  return project("artistic-vibrancy", {
    resources: [web],
  });
});
