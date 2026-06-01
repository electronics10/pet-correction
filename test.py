from mcgpu_backend.runner import Runner

run = Runner()
result = run("data/run_0", on_existing="skip")
print(result.sinogram_trues, result.wall_time_s)