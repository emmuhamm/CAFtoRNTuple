x = CAFFile(args.nominal, args.name)
y = CAFFile(args.up, args.name)


"rec.mc.nnu"

"rec.common.ixn.ndlp"
rec.mc.nu.id

track_x = x.get_data("rec.mc.nu.E")
track_x = x.get_data("rec.mc.nu")
track_y = x.get_data("rec.common.ixn.dlp")


also check in rec.mc.id and the .E as well


track_types = x.get_data(x._track_type)
for value in track_x[1]:
    print(value)

for value in track_y[1]:
    print(value)
print(ak.almost_equal(track_x[1], track_y[1], rtol=0, atol=0))

    Got another job. This time we want to look at thhings under rec.nd.lar.dlp... 
I want a plot of just the histograms of the charge and energies individually

then i want to make new branches under dlp.total_charge and dlp.total_energy 
this should just be the sum of the charge and energies respectively. 

plotted at 95% percentile

for ixn, (xs, ys) in enumerate(zip(track_x[1], track_y[1], strict=True)):
    no for particle, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if x != y:
            print(f"Interaction {ixn}, particle {particle}: {x} != {y}")



---

for event, (nominal, up) in enumerate(zip(track_x, track_y, strict=True)):
    same = np.array_equal(
        np.sort(ak.to_numpy(nominal)),
        np.sort(ak.to_numpy(up)),
    )
    print(f"Interaction {event}: {'EQUAL' if same else 'DIFFERENT'}")


from itertools import zip_longest
missing = object()
for event in (0, 3, 4):
    nominal = np.sort(ak.to_numpy(track_x[event]))
    up = np.sort(ak.to_numpy(track_y[event]))
    print(f"\nInteraction {event}: nominal={len(nominal)}, up={len(up)} interactions")
    for index, (a, b) in enumerate(
        zip_longest(nominal, up, fillvalue=missing)
    ):
        same = a is not missing and b is not missing and a == b
        if event == 0 or not same:
            a_text = "MISSING" if a is missing else f"{float(a):.17g}"
            b_text = "MISSING" if b is missing else f"{float(b):.17g}"
            print(
                f"  Sorted position {index}: nominal={a_text}, up={b_text}"
                f" -> {'EQUAL' if same else 'DIFFERENT'}"
            )


---

reconstructed_energy = x.get_data("rec.common.ixn.dlp.E")
true_energy = x.get_data("rec.mc.nu.E")
reconstructed_index = ak.local_index(reconstructed_energy, axis=1)
true_index = ak.local_index(true_energy, axis=1)
reconstructed_count = ak.num(reconstructed_energy, axis=1)
true_count = ak.num(true_energy, axis=1)

matched_reconstructed = reconstructed_energy[
    reconstructed_index < true_count
]
matched_true = true_energy[true_index < reconstructed_count]
valid = (
    np.isfinite(matched_reconstructed)
    & np.isfinite(matched_true)
    & (matched_true != 0)
)
fractional_difference = (
    matched_reconstructed - matched_true
) / matched_true


---

reconstructed_energy = x.get_data("rec.common.ixn.dlp.E")
truth_indices = x.get_data("rec.common.ixn.dlp.truth")
truth_overlap = x.get_data("rec.common.ixn.dlp.truthOverlap")
true_energy = x.get_data("rec.mc.nu.E")

best_match = ak.argmax(
    truth_overlap, axis=-1, keepdims=True, mask_identity=True
)
best_truth_index = ak.firsts(truth_indices[best_match], axis=-1)
best_truth_index = ak.mask(
    best_truth_index, best_truth_index < ak.num(true_energy, axis=1)
)
matched_true = ak.Array(
    [
        spill_energy[spill_truth_indices]
        for spill_energy, spill_truth_indices in zip(
            true_energy, best_truth_index
        )
    ]
)
valid = (
    np.isfinite(reconstructed_energy)
    & np.isfinite(matched_true)
    & (matched_true != 0)
)
valid = ak.fill_none(valid, False)
fractional_difference = (reconstructed_energy - matched_true) / matched_true