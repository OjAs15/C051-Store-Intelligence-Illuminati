from src.pipeline import refresh_v2_pipeline
if __name__=="__main__":
 r=refresh_v2_pipeline(); print("Output:",r["output_path"]); print([(x["area"],x["incremental_demand"]) for x in r["ranking"]])
