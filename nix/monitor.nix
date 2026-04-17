{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ai-inference;
  inherit (lib) mkIf;

  pythonEnv = pkgs.python3.withPackages (ps: [
    ps.prometheus-client
    ps.httpx
  ]);

  monitorPackage = pkgs.writeTextFile {
    name = "ai-inference-monitor.py";
    text = ''
      #!${pythonEnv}/bin/python3
      """
      AI Inference Monitor - Standalone metrics exporter
      Provides detailed metrics beyond what the gateway exports
      """
      import os
      import time
      import json
      import subprocess
      from prometheus_client import (
          Gauge, Counter, Histogram, CollectorRegistry, generate_latest,
          CONTENT_TYPE_LATEST
      )
      from prometheus_client.exposition import start_http_server

      import logging

      logging.basicConfig(
          level=logging.INFO,
          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      )
      logger = logging.getLogger("ai-inference-monitor")

      BACKEND_URL = os.getenv("BACKEND_URL", "${cfg.backend.url}")
      GATEWAY_URL = os.getenv("GATEWAY_URL", "http://${cfg.gateway.host}:${toString cfg.gateway.port}")

      NVIDIA_SMI_PATHS = [
          "/run/opengl-driver/bin/nvidia-smi",
          "/usr/bin/nvidia-smi",
          "${pkgs.linuxPackages.nvidia_x11}/bin/nvidia-smi",
      ]

      NVIDIA_SMI = None
      for path in NVIDIA_SMI_PATHS:
          if os.path.exists(path):
              NVIDIA_SMI = path
              logger.info(f"Found nvidia-smi at: {path}")
              break

      if not NVIDIA_SMI:
          logger.warning("nvidia-smi not found, GPU monitoring disabled")

      GPU_IDS = os.getenv("GPU_IDS", "0,1").split(",")

      gpu_vram_used = Gauge('ai_inference_gpu_vram_used_mb', 'VRAM used per GPU', ['gpu_id'])
      gpu_vram_total = Gauge('ai_inference_gpu_vram_total_mb', 'Total VRAM per GPU', ['gpu_id'])
      gpu_utilization = Gauge('ai_inference_gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
      gpu_temperature = Gauge('ai_inference_gpu_temperature_c', 'GPU temperature', ['gpu_id'])
      gpu_power_draw = Gauge('ai_inference_gpu_power_draw_w', 'GPU power draw', ['gpu_id'])

      backend_latency = Histogram('ai_inference_backend_latency_seconds', 'Backend request latency')
      backend_healthy = Gauge('ai_inference_backend_healthy', 'Backend health status')

      model_loaded = Gauge('ai_inference_model_loaded', 'Model loaded status', ['model'])

      def get_gpu_metrics():
          """Get GPU metrics from nvidia-smi"""
          if not NVIDIA_SMI:
              return

          try:
              result = subprocess.run([
                  NVIDIA_SMI,
                  '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                  '--format=csv,noheader,nounits'
              ], capture_output=True, text=True, timeout=5)

              if result.returncode != 0:
                  logger.warning(f"nvidia-smi returned non-zero exit code: {result.returncode}")
                  if result.stderr:
                      logger.debug(f"nvidia-smi stderr: {result.stderr}")
                  return

              for line in result.stdout.strip().split('\n'):
                  if not line:
                      continue
                  parts = [p.strip() for p in line.split(',')]
                  if len(parts) < 6:
                      continue

                  gpu_id, mem_used, mem_total, util, temp, power = parts[:6]

                  try:
                      gpu_vram_used.labels(gpu_id=gpu_id).set(float(mem_used))
                      gpu_vram_total.labels(gpu_id=gpu_id).set(float(mem_total))
                      gpu_utilization.labels(gpu_id=gpu_id).set(float(util))
                      gpu_temperature.labels(gpu_id=gpu_id).set(float(temp))
                      gpu_power_draw.labels(gpu_id=gpu_id).set(float(power))
                  except ValueError as e:
                      logger.warning(f"Failed to parse GPU metrics for GPU {gpu_id}: {e}")
                      continue

          except FileNotFoundError:
              logger.error(f"nvidia-smi not found at {NVIDIA_SMI}")
          except subprocess.TimeoutExpired:
              logger.warning("nvidia-smi command timed out")
          except Exception as e:
              logger.exception(f"Unexpected error getting GPU metrics: {e}")

      def check_backend_health():
          """Check backend health"""
          try:
              import httpx
              with httpx.Client(timeout=5.0) as client:
                  resp = client.get(f"{BACKEND_URL}/v1/models")
                  backend_healthy.set(1 if resp.status_code == 200 else 0)

                  if resp.status_code == 200:
                      data = resp.json()
                      models = data.get("data", [])
                      for m in ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-35b-a3b"]:
                          model_loaded.labels(model=m).set(0)
                      for m in models:
                          model_id = m.get("id", "")
                          for known in ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-35b-a3b"]:
                              if known in model_id:
                                  model_loaded.labels(model=known).set(1)
                                  logger.debug(f"Model {known} is loaded")

          except Exception as e:
              backend_healthy.set(0)
              logger.error(f"Backend health check failed: {e}")

      def main():
          """Main monitoring loop"""
          logger.info(f"Starting AI Inference Monitor on port ${toString cfg.monitoring.port}")
          logger.info(f"Backend: {BACKEND_URL}")
          logger.info(f"Gateway: {GATEWAY_URL}")

          start_http_server(int(os.getenv("METRICS_PORT", "${toString cfg.monitoring.port}")))

          get_gpu_metrics()
          check_backend_health()

          logger.info("Starting metrics collection loop (15s interval)")
          while True:
              time.sleep(15)
              get_gpu_metrics()
              check_backend_health()

      if __name__ == "__main__":
          main()
    '';
    executable = true;
  };
in {
  config = mkIf (cfg.enable && cfg.monitoring.enable) {
    systemd.services.ai-inference-monitor = {
      description = "AI Inference Metrics Monitor";
      after = ["network.target"];
      wantedBy = ["multi-user.target"];

      environment = {
        BACKEND_URL = cfg.backend.url;
        GATEWAY_URL = "http://${cfg.gateway.host}:${toString cfg.gateway.port}";
        METRICS_PORT = toString cfg.monitoring.port;
      };

      serviceConfig = {
        ExecStart = monitorPackage;

        Restart = "on-failure";
        RestartSec = "10s";

        User = "root";
        Group = "root";
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectHome = true;

        StandardOutput = "journal";
        StandardError = "journal";
        SyslogIdentifier = "ai-monitor";
      };
    };

  };
}
