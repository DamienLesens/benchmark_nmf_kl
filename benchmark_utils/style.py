import matplotlib.pyplot as plt


def parse_solver_string(s):
    # split name and options
    name, opts = s.split("[", 1)
    opts = opts.rstrip("]")

    # parse key=value pairs
    result = {}
    for item in opts.split(","):
        key, value = item.split("=", 1)

        # convert types
        value = value.strip()
        if value == "None":
            value = None
        elif value == "True":
            value = True
        elif value == "False":
            value = False
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # keep as string

        result[key.strip()] = value

    return name, result

def get_style(solver):
    """
    This function is used to modify solver labels across all custom plots
    """
    style = {}
    name,options = parse_solver_string(solver)
    match name:
        case "mu":
            style["label"] = "MU"
        case "fpa":
            style["label"] = "FPA"
        case "hals":
            style["label"] = "HALS"
        case "scalar_newton":
            style["label"] = options['method']
        case "som":
            style["label"] = options['method']
        case "newton":
            if options["descent"]:
                style["label"] = "KL-HALS descent"
            else:
                style["label"] = "KL-HALS"
        case _:
            style["label"] = name

    CMAP = plt.get_cmap('tab20')
    COLORS = [CMAP(i) for i in range(CMAP.N)]
    COLORS = COLORS[::2] + COLORS[1::2]
        
    match style["label"]:
        case "MU":
            style["color"]=COLORS[2]
            style["marker"]=2
        case "MU_Burg":
            style["color"]=COLORS[9]
            style["marker"]=9
        case "FPA":
            style["color"]=COLORS[0]
            style["marker"]=0
        case "ADMM":
            style["color"]=COLORS[8]
            style["marker"]=8
        case "AMUSOM":
            style["color"]=COLORS[6]
            style["marker"]=6
        case "AmSOM":
            style["color"]=COLORS[7]
            style["marker"]=7
        case "SN":
            style["color"]=COLORS[5]
            style["marker"]=5
        case "CCD":
            style["color"]=COLORS[4]
            style["marker"]=4
        case "KL-HALS":
            style["color"]=COLORS[3]
            style["marker"]=3
        case "KL-HALS descent":
            style["color"]=COLORS[13]
            style["marker"]=13
        case "HALS":
            style["color"]=COLORS[1]
            style["marker"]=1
    
    return style


