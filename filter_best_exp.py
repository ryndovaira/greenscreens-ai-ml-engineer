import json
import os


def find_best_exp() -> dict[str:float]:
    best_exp = {}
    for file in os.listdir("experiments"):
        # open if exists
        if not os.path.exists(f"experiments/{file}/validation_results.json"):
            continue
        with open(f"experiments/{file}/validation_results.json") as f:
            exp = json.load(f)
        mape = exp["MAPE"]
        if mape < 15:
            best_exp[file] = mape

    return best_exp


if __name__ == "__main__":
    best_exp = find_best_exp()
    best_exp = dict(sorted(best_exp.items(), key=lambda item: item[1]))
    print(json.dumps(best_exp, indent=4))
    for exp in best_exp:
        os.system(
            f"xcopy experiments\\{exp} best_experiments\\exp_{best_exp[exp]}_{exp} /E /I /H /Y"
        )
    with open("best_experiments/best_experiments.json", "w") as f:
        json.dump(best_exp, f, indent=4)
