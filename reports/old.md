# Algoritmo secuencial

```c++
for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        // dx, dy, dz = distancia entre bi y bj
        // distancia euclidiana
        distSqr = dx * dx + dy * dy + dz * dz + NEAR_ZERO;
        // recíproco de la distancia
        invDist = rsqrtf(distSqr);
        float bi mass = (bi.special ? *special_mass : *mass);
        float bj mass = (bj.special ? *special_mass : *mass); 
        // cálculo de F
```

# Algoritmo paralelo
Se define un kernel que realiza el cálculo de la fuerza neta sobre una partícula
utilizando la misma idea del algoritmo secuencial. Utilizando CUDA, primero se copian
los datos al dispositivo. Luego se lanza el kernel y se transfieren los datos de vuelta.

```c++
extern "C" __global__ void updateBodies(...) {
    // cada thread maneja una partícula
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // cálculo de F ...
}
cudaDeviceSynchronize();
// recuperar las posiciones y velocidades actualizadas
cudaMemcpy(h_bodies, d_bodies, size, cudaMemcpyDeviceToHost);
// liberar memoria
cudaFree(d_bodies);
cuCtxDestroy(context);
```