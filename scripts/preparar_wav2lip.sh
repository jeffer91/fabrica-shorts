#!/usr/bin/env bash
# Descarga Wav2Lip y sus modelos, y lo adapta a librerias nuevas (librosa >= 0.10, torch >= 2.6).
# Si algo falla, el video se arma igual con el avatar de imagenes.
set -euo pipefail
DIR="${1:-wav2lip}"
if [ ! -d "$DIR/.git" ]; then
  git clone -q --depth 1 https://github.com/Rudrabha/Wav2Lip "$DIR"
fi
bajar() {  # bajar <destino> <url>
  [ -s "$1" ] && [ "$(stat -c %s "$1")" -gt 1000000 ] && return 0
  curl -sfL --retry 4 --retry-all-errors -o "$1" "$2"
}
bajar "$DIR/checkpoints/wav2lip_gan.pth" https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth
bajar "$DIR/face_detection/detection/sfd/s3fd.pth" https://github.com/justinjohn0306/Wav2Lip/releases/download/models/s3fd.pth
cd "$DIR"
sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,/' audio.py
sed -i 's/torch.load(checkpoint_path)/torch.load(checkpoint_path, weights_only=False)/; s/torch.load(checkpoint_path,$/torch.load(checkpoint_path, weights_only=False,/' inference.py
sed -i 's/torch.load(path_to_detector)/torch.load(path_to_detector, weights_only=False)/' face_detection/detection/sfd/sfd_detector.py
sed -i 's/dtype=np.int)/dtype=int)/' face_detection/utils.py
# Cache de caras: tu clip no cambia, asi que la deteccion (lo mas lento) se hace una sola vez.
python - <<'PY'
p = "inference.py"
s = open(p).read()
if "W2L_CACHE" not in s:
    s = s.replace("def face_detect(images):\n",
        "def face_detect(images):\n"
        "\tcache = os.environ.get('W2L_CACHE')\n"
        "\tif cache and os.path.isfile(cache):\n"
        "\t\tguardadas = np.load(cache)\n"
        "\t\tif len(guardadas) >= len(images):\n"
        "\t\t\tprint('Caras desde cache')\n"
        "\t\t\treturn [[im[y1: y2, x1:x2], (y1, y2, x1, x2)] for im, (x1, y1, x2, y2) in zip(images, guardadas[:len(images)])]\n"
        "\treturn _face_detect(images)\n\n"
        "def _face_detect(images):\n", 1)
    s = s.replace("\tboxes = np.array(results)\n",
        "\tboxes = np.array(results)\n"
        "\tif os.environ.get('W2L_CACHE'): np.save(os.environ['W2L_CACHE'], boxes)\n", 1)
    open(p, "w").write(s)
PY
mkdir -p temp results
echo "Wav2Lip listo"
