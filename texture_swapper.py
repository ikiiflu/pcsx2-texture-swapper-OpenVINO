#!/usr/bin/env python3
"""
Texture Swapper com IA — PCSX2
Modelo  : DreamShaper 8 via OpenVINO (optimum.intel) + BLIP captioning
Hardware: CPU-only

Estrutura esperada:
  <pasta_do_jogo>/          <- usuário seleciona esta pasta diretamente
      dumps/                <- texturas originais (PCSX2 dumpa aqui)
      replacements/         <- texturas geradas (script salva aqui)
"""

import json, random, sys, threading, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime

_blip_processor = None
_blip_model     = None
_pipeline_i2i   = None   # img2img  — usado no modo com contexto (BLIP + DreamShaper)
_pipeline_t2i   = None   # txt2img  — usado na Geração sem Contexto (sem BLIP)

SCRIPT_DIR      = Path(sys.argv[0]).resolve().parent
PROMPTS_FILE    = SCRIPT_DIR / "prompts.json"
RAND_WORDS_FILE = SCRIPT_DIR / "random_words.json"
TEXTURE_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"}

DEFAULT_TEXTURE_CONTEXT = (
    "seamless flat video game texture map, high quality albedo material, "
    "top-down orthographic view, game asset"
)
DEFAULT_NEGATIVE_PROMPT = (
    "3d scene, perspective, shadows, room, lighting, character, UI, "
    "text, watermark, depth of field, borders"
)

BG="1e2228";BG_DARK="16191e";BG_MID="252a33";BG_INPUT="1a1f27"
BORDER="3a7bd5";FG="e8ecf0";FG_DIM="7a8a9a";GREEN="27ae3f";GREEN_DK="1e8c30"
RED="c0392b";RED_DK="922b21";ENTRY_FG="d0dce8";SEP="2a3340"
LOG_BG="141820";LOG_FG="7aaa78";C_INFO="5aade8";C_WARN="e8a53a"
C_ERR="e85a5a";C_OK="5ae870";C_WORD="d4a0f0"
def h(c): return "#"+c
BG=h(BG);BG_DARK=h(BG_DARK);BG_MID=h(BG_MID);BG_INPUT=h(BG_INPUT)
BORDER=h(BORDER);FG=h(FG);FG_DIM=h(FG_DIM);GREEN=h(GREEN);GREEN_DK=h(GREEN_DK)
RED=h(RED);RED_DK=h(RED_DK);ENTRY_FG=h(ENTRY_FG);SEP=h(SEP)
LOG_BG=h(LOG_BG);LOG_FG=h(LOG_FG);C_INFO=h(C_INFO);C_WARN=h(C_WARN)
C_ERR=h(C_ERR);C_OK=h(C_OK);C_WORD=h(C_WORD)

F_H1=("Segoe UI",15,"bold");F_LBL=("Segoe UI",9);F_SMALL=("Segoe UI",8)
F_MONO=("Consolas",9);F_BTN=("Segoe UI",10,"bold");F_PNL=("Segoe UI",9,"bold")

def load_json(path, fallback):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] {path}: {e}")
    return fallback

def load_models_i2i(log_fn):
    """Carrega BLIP + DreamShaper img2img (modo com contexto)."""
    global _blip_processor,_blip_model,_pipeline_i2i
    if _pipeline_i2i is not None:
        return True
    try:
        from transformers import BlipProcessor,BlipForConditionalGeneration
        from optimum.intel import OVStableDiffusionImg2ImgPipeline
        from diffusers import LCMScheduler
        log_fn("[1/2] Carregando BLIP (visão / captioning)...","info")
        _blip_processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        log_fn("[2/2] Carregando DreamShaper 8 img2img via OpenVINO (1ª vez ~5 min)...","warn")
        pipe=OVStableDiffusionImg2ImgPipeline.from_pretrained("Lykon/dreamshaper-8",export=True,device="CPU")
        pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
        def _ns(images,**kw): return images,[False]*len(images)
        pipe.safety_checker=_ns
        _pipeline_i2i=pipe
        log_fn("Modelo img2img carregado!","ok"); return True
    except Exception as e:
        log_fn(f"Erro ao carregar modelo img2img: {e}","err"); return False

def load_models_t2i(log_fn):
    """Carrega DreamShaper txt2img (Geração sem Contexto — sem BLIP)."""
    global _pipeline_t2i
    if _pipeline_t2i is not None:
        return True
    try:
        from optimum.intel import OVStableDiffusionPipeline
        from diffusers import LCMScheduler
        log_fn("Carregando DreamShaper 8 txt2img via OpenVINO (1ª vez ~5 min)...","warn")
        pipe=OVStableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8",export=True,device="CPU")
        pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
        def _ns(images,**kw): return images,[False]*len(images)
        pipe.safety_checker=_ns
        _pipeline_t2i=pipe
        log_fn("Modelo txt2img carregado!","ok"); return True
    except Exception as e:
        log_fn(f"Erro ao carregar modelo txt2img: {e}","err"); return False

# ── img2img com BLIP — modo com contexto ──────────────────────────────────────
def process_img2img(src,dst,theme_prompt,negative_prompt,steps,strength,guidance,log_fn,stop_ev):
    """BLIP descreve a textura original → DreamShaper img2img transforma."""
    if stop_ev.is_set(): return "stopped"
    try:
        import torch; from PIL import Image
        original=Image.open(src).convert("RGBA")
        orig_w,orig_h=original.size
        r,g,b,alpha=original.split()
        rgb_img=Image.merge("RGB",(r,g,b))
        # BLIP descreve a textura
        inputs=_blip_processor(rgb_img,return_tensors="pt")
        with torch.no_grad(): out=_blip_model.generate(**inputs)
        description=_blip_processor.decode(out[0],skip_special_tokens=True)
        log_fn(f"  BLIP: {description}","info")
        final_prompt=f"{DEFAULT_TEXTURE_CONTEXT}, {theme_prompt}, showing {description}"
        s=final_prompt[:95]+("..." if len(final_prompt)>95 else "")
        log_fn(f"  Prompt: {s}","info")
        result=_pipeline_i2i(prompt=final_prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=rgb_img.resize((512,512)),strength=strength,
            num_inference_steps=steps,guidance_scale=guidance).images[0]
        new_w,new_h=orig_w*2,orig_h*2
        final=result.resize((new_w,new_h),Image.LANCZOS).convert("RGBA")
        alpha_r=alpha.resize((new_w,new_h),Image.NEAREST)
        final.putalpha(alpha_r)
        dst_png=dst.with_suffix(".png")
        dst_png.parent.mkdir(parents=True,exist_ok=True)
        final.save(str(dst_png)); return "ok"
    except Exception as e:
        log_fn(f"  Erro: {e}","err"); return "error"

# ── txt2img puro — Geração sem Contexto (sem BLIP, sem imagem de entrada) ─────
def process_txt2img(dst,word_prompt,steps,guidance,log_fn,stop_ev):
    """Gera textura a partir de uma palavra/prompt puro, sem ler a imagem original."""
    if stop_ev.is_set(): return "stopped"
    try:
        from PIL import Image
        log_fn(f"  Prompt txt2img: \"{word_prompt}\"","info")
        result=_pipeline_t2i(prompt=word_prompt,
            negative_prompt=None,
            num_inference_steps=steps,
            guidance_scale=guidance).images[0]
        # Salva em 512x512 (tamanho nativo do modelo)
        final=result.convert("RGBA")
        dst_png=dst.with_suffix(".png")
        dst_png.parent.mkdir(parents=True,exist_ok=True)
        final.save(str(dst_png)); return "ok"
    except Exception as e:
        log_fn(f"  Erro: {e}","err"); return "error"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self._running=False; self._stop_ev=threading.Event()
        self._prompts=load_json(PROMPTS_FILE,[])
        self._words=load_json(RAND_WORDS_FILE,["abstract"])
        # Configura cores do dropdown nativo ANTES de criar widgets
        self.option_add("*TCombobox*Listbox.background", BG_INPUT)
        self.option_add("*TCombobox*Listbox.foreground", ENTRY_FG)
        self.option_add("*TCombobox*Listbox.selectBackground", BORDER)
        self.option_add("*TCombobox*Listbox.selectForeground","#ffffff")
        self.option_add("*TCombobox*Listbox.font", F_LBL)
        self.title("Texture Swapper com IA")
        self.configure(bg=BG_DARK)
        self.resizable(False,False)
        self._setup_style()
        self._build_ui()
        self.update_idletasks()
        self.geometry(f"680x{self.winfo_reqheight()+6}")
        self._log("Sistema iniciado. Configure os campos e clique em INICIAR.","info")

    def _setup_style(self):
        s=ttk.Style(self); s.theme_use("clam")
        s.configure("TNotebook",background=BG,borderwidth=0,tabmargins=[0,0,0,0])
        s.configure("TNotebook.Tab",background="#1c2028",foreground=FG_DIM,padding=[14,5],font=F_LBL)
        s.map("TNotebook.Tab",background=[("selected",BG_MID)],foreground=[("selected",BORDER)])

    def _ekw(self):
        return dict(bg=BG_INPUT,fg=ENTRY_FG,relief="flat",font=F_LBL,
                    insertbackground=FG,highlightthickness=1,
                    highlightbackground="#2e3a4a",highlightcolor=BORDER)

    def _bsm(self):
        return dict(bg=BG_MID,fg=FG,relief="flat",font=F_LBL,
                    cursor="hand2",activebackground=BORDER,activeforeground="#fff")

    def _build_ui(self):
        bar=tk.Frame(self,bg="#0f1218",height=26); bar.pack(fill="x")
        tk.Label(bar,text="  Texture Swapper com IA  —  PCSX2",
                 bg="#0f1218",fg=FG_DIM,font=F_SMALL).pack(side="left",pady=4)

        body=tk.Frame(self,bg=BG,padx=16,pady=10); body.pack(fill="both",expand=True)
        tk.Label(body,text="Texture Swapper com IA",bg=BG,fg=FG,font=F_H1).pack(anchor="w",pady=(2,10))

        # Pasta do jogo
        tk.Label(body,text="Pasta do Jogo:",bg=BG,fg=FG,font=F_LBL).pack(anchor="w")
        rp=tk.Frame(body,bg=BG); rp.pack(fill="x",pady=(2,2))
        self._game_var=tk.StringVar(value=r"")
        tk.Entry(rp,textvariable=self._game_var,**self._ekw()).pack(side="left",fill="x",expand=True,ipady=5,ipadx=6)
        tk.Button(rp,text="...",**self._bsm(),command=self._browse).pack(side="left",padx=(4,0),ipady=4,ipadx=6)
        tk.Label(body,text="Selecione a pasta do jogo que contém dumps/ e replacements/",
                 bg=BG,fg=FG_DIM,font=F_SMALL).pack(anchor="w",pady=(2,6))

        # Contador
        rc=tk.Frame(body,bg=BG); rc.pack(fill="x",pady=(0,8))
        tk.Label(rc,text="Texturas em dumps/:",bg=BG,fg=FG,font=F_LBL).pack(side="left")
        self._count_lbl=tk.Label(rc,text="—",bg=BG,fg=BORDER,font=("Consolas",11,"bold"))
        self._count_lbl.pack(side="left",padx=(6,6))
        tk.Button(rc,text="↺  Contar",**self._bsm(),command=self._refresh_count).pack(side="left",ipady=2,ipadx=8)

        # Notebook
        nb=ttk.Notebook(body); nb.pack(fill="x",pady=(2,8))
        tp=tk.Frame(nb,bg=BG_MID,padx=12,pady=10)
        tm=tk.Frame(nb,bg=BG_MID,padx=12,pady=10)
        tq=tk.Frame(nb,bg=BG_MID,padx=12,pady=10)
        nb.add(tp,text="   Prompt   ")
        nb.add(tm,text="   Modo de Geração   ")
        nb.add(tq,text="   Parâmetros   ")
        self._build_tab_prompt(tp)
        self._build_tab_mode(tm)
        self._build_tab_params(tq)

        # Botões
        br=tk.Frame(body,bg=BG); br.pack(fill="x",pady=(0,6))
        self._start_btn=tk.Button(br,text="INICIAR SWAPPER",bg=GREEN,fg="#fff",relief="flat",
            font=F_BTN,cursor="hand2",activebackground=GREEN_DK,activeforeground="#fff",
            height=2,command=self._toggle_run)
        self._start_btn.pack(side="left",fill="x",expand=True,padx=(0,5))
        tk.Button(br,text="EXCLUIR REPLACEMENTS",bg="#2a1515",fg=C_ERR,relief="flat",
            font=F_BTN,cursor="hand2",activebackground=RED_DK,activeforeground="#fff",
            height=2,command=self._delete_replacements).pack(side="left",fill="x",expand=True)

        # Log
        lf=tk.Frame(body,bg=LOG_BG); lf.pack(fill="both",expand=True)
        self._log_txt=tk.Text(lf,bg=LOG_BG,fg=LOG_FG,font=F_MONO,relief="flat",
            state="disabled",wrap="word",height=8,padx=8,pady=6,
            insertbackground=FG,selectbackground=BORDER)
        sb=tk.Scrollbar(lf,command=self._log_txt.yview,bg=BG_DARK,
            troughcolor=BG_DARK,bd=0,activebackground=BORDER)
        sb.pack(side="right",fill="y"); self._log_txt.pack(fill="both",expand=True)
        self._log_txt.configure(yscrollcommand=sb.set)
        for tag,col in [("ok",C_OK),("info",C_INFO),("warn",C_WARN),("err",C_ERR),("word",C_WORD)]:
            self._log_txt.tag_config(tag,foreground=col)

    def _build_tab_prompt(self,p):
        tk.Label(p,text="Tema pré-definido (prompts.json):",bg=BG_MID,fg=FG,font=F_LBL).pack(anchor="w")

        if self._prompts:
            names=[x.get("name","") for x in self._prompts]
        else:
            names=["— prompts.json não encontrado —"]

        self._preset_menu_var=tk.StringVar(value="Selecione um tema...")
        # tk.OptionMenu: dark-theme confiável, sem problemas de listbox
        om=tk.OptionMenu(p,self._preset_menu_var,*names,command=self._on_preset_select)
        om.config(bg=BG_INPUT,fg=ENTRY_FG,relief="flat",activebackground=BORDER,
                  activeforeground="#fff",font=F_LBL,cursor="hand2",
                  highlightthickness=1,highlightbackground="#2e3a4a",
                  indicatoron=True,anchor="w",width=56)
        om["menu"].config(bg=BG_INPUT,fg=ENTRY_FG,activebackground=BORDER,
                          activeforeground="#fff",font=F_LBL,relief="flat",bd=0)
        om.pack(fill="x",pady=(3,10),ipady=3)

        tk.Label(p,text="Prompt (tema / estilo desejado):",bg=BG_MID,fg=FG,font=F_LBL).pack(anchor="w")
        self._prompt_var=tk.StringVar(value="pixel art, 16-bit retro game texture, dithering")
        tk.Entry(p,textvariable=self._prompt_var,**self._ekw()).pack(fill="x",ipady=5,ipadx=6,pady=(2,10))

        rn=tk.Frame(p,bg=BG_MID); rn.pack(fill="x",pady=(0,3))
        tk.Label(rn,text="Negative Prompt:",bg=BG_MID,fg=FG,font=F_LBL).pack(side="left")
        self._use_def_neg=tk.BooleanVar(value=True)
        tk.Checkbutton(rn,text="Usar padrão",variable=self._use_def_neg,bg=BG_MID,fg=FG_DIM,
            activebackground=BG_MID,activeforeground=FG,selectcolor=BG_INPUT,
            font=F_SMALL,cursor="hand2",command=self._toggle_neg).pack(side="right")
        self._neg_var=tk.StringVar(value=DEFAULT_NEGATIVE_PROMPT)
        self._neg_entry=tk.Entry(p,textvariable=self._neg_var,state="disabled",**self._ekw())
        self._neg_entry.pack(fill="x",ipady=5,ipadx=6,pady=(2,0))
        tk.Label(p,text="Dica: o Negative Prompt é ignorado automaticamente na Geração sem Contexto.",
                 bg=BG_MID,fg=FG_DIM,font=F_SMALL).pack(anchor="w",pady=(5,0))

    def _build_tab_mode(self,p):
        self._mode_var=tk.StringVar(value="total")
        rb_kw=dict(bg=BG_MID,fg=FG,activebackground=BG_MID,activeforeground=FG,
                   selectcolor=BG_INPUT,font=F_PNL,cursor="hand2",
                   variable=self._mode_var,command=self._on_mode_change)

        tk.Radiobutton(p,text="Geração Total",value="total",**rb_kw).pack(anchor="w")
        tk.Label(p,text="   Processa todas as texturas da pasta dumps/.",bg=BG_MID,fg=FG_DIM,font=F_SMALL).pack(anchor="w")
        tk.Frame(p,bg=SEP,height=1).pack(fill="x",pady=8)
        tk.Radiobutton(p,text="Geração Parcial",value="parcial",**rb_kw).pack(anchor="w")
        tk.Label(p,text="   Escolhe um subconjunto aleatório de texturas.",bg=BG_MID,fg=FG_DIM,font=F_SMALL).pack(anchor="w")

        self._pframe=tk.Frame(p,bg=BG_MID); self._pframe.pack(fill="x",padx=22,pady=(6,0))
        rn=tk.Frame(self._pframe,bg=BG_MID); rn.pack(anchor="w",pady=(0,4))
        self._lbl_n=tk.Label(rn,text="Número de imagens:",bg=BG_MID,fg=FG_DIM,font=F_LBL); self._lbl_n.pack(side="left")
        self._parcial_n=tk.IntVar(value=10)
        self._spin_n=tk.Spinbox(rn,from_=1,to=9999,textvariable=self._parcial_n,width=7,
            bg=BG_INPUT,fg=ENTRY_FG,relief="flat",font=F_MONO,insertbackground=FG,
            buttonbackground=BG_MID,disabledbackground=BG_INPUT,disabledforeground=FG_DIM)
        self._spin_n.pack(side="left",padx=(6,0))
        self._use_half=tk.BooleanVar(value=False)
        self._chk_half=tk.Checkbutton(self._pframe,text="OU: metade das texturas  +  N extras aleatórias:",
            variable=self._use_half,bg=BG_MID,fg=FG_DIM,activebackground=BG_MID,activeforeground=FG,
            selectcolor=BG_INPUT,font=F_LBL,cursor="hand2",command=self._on_mode_change)
        self._chk_half.pack(anchor="w")
        rh=tk.Frame(self._pframe,bg=BG_MID); rh.pack(anchor="w",padx=24,pady=(2,0))
        self._lbl_extra=tk.Label(rh,text="N extras:",bg=BG_MID,fg=FG_DIM,font=F_LBL); self._lbl_extra.pack(side="left")
        self._extra_n=tk.IntVar(value=5)
        self._spin_extra=tk.Spinbox(rh,from_=0,to=9999,textvariable=self._extra_n,width=7,
            bg=BG_INPUT,fg=ENTRY_FG,relief="flat",font=F_MONO,insertbackground=FG,
            buttonbackground=BG_MID,disabledbackground=BG_INPUT,disabledforeground=FG_DIM)
        self._spin_extra.pack(side="left",padx=(6,0))

        tk.Frame(p,bg=SEP,height=1).pack(fill="x",pady=8)
        self._sem_ctx=tk.BooleanVar(value=False)
        tk.Checkbutton(p,text="Geração sem Contexto  (combinável com Total ou Parcial)",
            variable=self._sem_ctx,bg=BG_MID,fg=FG,activebackground=BG_MID,activeforeground=FG,
            selectcolor=BG_INPUT,font=F_PNL,cursor="hand2").pack(anchor="w")
        tk.Label(p,text=(
            "   Ignora TEXTURE_CONTEXT e Negative Prompt.\n"
            "   Cada textura recebe uma palavra diferente sorteada de random_words.json."),
            bg=BG_MID,fg=FG_DIM,font=F_SMALL,justify="left").pack(anchor="w",pady=(2,0))
        self._on_mode_change()

    def _build_tab_params(self,p):
        self._steps_v=tk.IntVar(value=4)
        self._strength_v=tk.DoubleVar(value=0.80)
        self._guidance_v=tk.DoubleVar(value=2.5)
        def slider(label,hint,var,lo,hi,res,is_int=False):
            tk.Label(p,text=label,bg=BG_MID,fg=FG,font=F_LBL).pack(anchor="w",pady=(8,0))
            tk.Label(p,text=hint,bg=BG_MID,fg=FG_DIM,font=F_SMALL).pack(anchor="w")
            r=tk.Frame(p,bg=BG_MID); r.pack(fill="x",pady=(2,0))
            sl=tk.Scale(r,from_=lo,to=hi,resolution=res,orient="horizontal",variable=var,
                bg=BG_MID,fg=FG,troughcolor=BG_INPUT,activebackground=BORDER,
                highlightthickness=0,bd=0,showvalue=False)
            sl.pack(side="left",fill="x",expand=True)
            fmt="{:.0f}" if is_int else "{:.2f}"
            lbl=tk.Label(r,text=fmt.format(var.get()),bg=BG_MID,fg=BORDER,font=F_MONO,width=6)
            lbl.pack(side="left",padx=(8,0))
            var.trace_add("write",lambda *_:lbl.config(text=fmt.format(var.get())))
        slider("STEPS  —  Passos de inferência","Ideal: 4 (LCM). 1=rápido, 8=qualidade.",self._steps_v,1,12,1,True)
        slider("STRENGTH  —  Intensidade da mudança","0.0=sem mudança | 0.8=recomendado | 1.0=total.",self._strength_v,0.0,1.0,0.01)
        slider("GUIDANCE SCALE  —  Aderência ao prompt","Mantenha ~2.5 com LCM. Valores altos podem distorcer.",self._guidance_v,0.5,15.0,0.1)
        tk.Frame(p,bg=SEP,height=1).pack(fill="x",pady=10)
        tk.Label(p,text="Recomendado para CPU:  STEPS=4  |  STRENGTH=0.80  |  GUIDANCE=2.5",
                 bg=BG_MID,fg=FG_DIM,font=F_SMALL).pack(anchor="w")

    # Callbacks
    def _on_preset_select(self,name):
        for pr in self._prompts:
            if pr.get("name")==name:
                self._prompt_var.set(pr.get("prompt",""))
                neg=pr.get("negative","")
                if neg:
                    self._use_def_neg.set(False); self._neg_entry.config(state="normal"); self._neg_var.set(neg)
                else:
                    self._use_def_neg.set(True); self._neg_var.set(DEFAULT_NEGATIVE_PROMPT); self._neg_entry.config(state="disabled")
                self._log(f"Tema aplicado: {name}","info"); break

    def _on_mode_change(self,*_):
        is_p=self._mode_var.get()=="parcial"; hc=self._use_half.get()
        sm="normal" if is_p else "disabled"; se="normal" if (is_p and hc) else "disabled"
        for w in [self._lbl_n,self._spin_n,self._chk_half,self._lbl_extra]:
            try: w.config(state=sm)
            except tk.TclError: pass
        try: self._spin_extra.config(state=se)
        except tk.TclError: pass

    def _toggle_neg(self):
        if self._use_def_neg.get():
            self._neg_var.set(DEFAULT_NEGATIVE_PROMPT); self._neg_entry.config(state="disabled")
        else:
            self._neg_entry.config(state="normal")

    def _browse(self):
        path=filedialog.askdirectory(title="Selecione a pasta DO JOGO (ex: SLUS-21728)")
        if path:
            self._game_var.set(path); self._refresh_count()

    def _refresh_count(self):
        d=self._dump_dir()
        if d is None: self._count_lbl.config(text="—"); return
        if not d.exists(): self._count_lbl.config(text="0  (dumps/ não existe)"); return
        try:
            n=sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in TEXTURE_EXTS)
            self._count_lbl.config(text=str(n))
        except Exception as e:
            self._count_lbl.config(text="erro"); self._log(f"Erro ao contar: {e}","err")

    def _dump_dir(self):
        g=self._game_var.get().strip()
        return Path(g)/"dumps" if g else None

    def _out_dir(self):
        g=self._game_var.get().strip()
        return Path(g)/"replacements" if g else None

    def _delete_replacements(self):
        out=self._out_dir()
        if not out or not out.exists():
            messagebox.showinfo("Excluir",f"Pasta não encontrada:\n{out}"); return
        imgs=[f for f in out.rglob("*") if f.is_file() and f.suffix.lower() in TEXTURE_EXTS]
        if not imgs:
            messagebox.showinfo("Excluir","Nenhuma imagem em replacements/."); return
        if not messagebox.askyesno("Confirmar exclusão",
            f"Excluir {len(imgs)} imagem(ns) de:\n{out}\n\nEsta ação não pode ser desfeita!"): return
        deleted=0
        for f in imgs:
            try: f.unlink(); deleted+=1
            except Exception as e: self._log(f"Erro ao excluir {f.name}: {e}","err")
        self._log(f"{deleted} imagem(ns) excluída(s) de replacements/.","warn")

    def _log(self,msg,level="ok"):
        ts=datetime.now().strftime("%H:%M:%S")
        self._log_txt.configure(state="normal")
        self._log_txt.insert("end",f"[{ts}] {msg}\n",level)
        self._log_txt.configure(state="disabled"); self._log_txt.see("end")

    def _log_safe(self,msg,level="ok"):
        self.after(0,lambda m=msg,lv=level:self._log(m,lv))

    def _toggle_run(self):
        if self._running:
            self._stop_ev.set(); self._log("Parando após a textura atual...","warn")
        else: self._start()

    def _start(self):
        dump=self._dump_dir()
        if not dump or not dump.exists():
            messagebox.showerror("Pasta não encontrada",
                f"A pasta dumps/ não existe:\n{dump}\n\n"
                "Estrutura esperada dentro da pasta do jogo:\n"
                "  dumps/        ← texturas originais\n"
                "  replacements/ ← serão salvas aqui"); return
        if not self._sem_ctx.get() and not self._prompt_var.get().strip():
            messagebox.showerror("Erro","O campo Prompt está vazio!"); return
        self._running=True; self._stop_ev.clear()
        self._start_btn.config(text="■  PARAR",bg=RED,activebackground=RED_DK)
        threading.Thread(target=self._worker,daemon=True).start()

    def _worker(self):
        try:
            dump_dir=self._dump_dir(); out_dir=self._out_dir()
            mode=self._mode_var.get(); sem_ctx=self._sem_ctx.get()

            # Carrega apenas o pipeline necessário para o modo escolhido
            if sem_ctx:
                if not load_models_t2i(self._log_safe): return
            else:
                if not load_models_i2i(self._log_safe): return

            all_files=sorted([f for f in dump_dir.iterdir()
                              if f.is_file() and f.suffix.lower() in TEXTURE_EXTS])
            if not all_files:
                self._log_safe("Nenhuma textura encontrada em dumps/!","err"); return

            if mode=="total":
                selected=list(all_files)
            else:
                if self._use_half.get():
                    half=len(all_files)//2
                    base=random.sample(all_files,min(half,len(all_files)))
                    pool=[f for f in all_files if f not in set(base)]
                    extra=random.sample(pool,min(self._extra_n.get(),len(pool)))
                    selected=base+extra; random.shuffle(selected)
                else:
                    n=self._parcial_n.get()
                    selected=random.sample(all_files,min(n,len(all_files)))

            total=len(selected)
            ml=("sem contexto + " if sem_ctx else "")+mode
            self._log_safe(f"Modo: {ml}  |  {total} de {len(all_files)} textura(s)","info")

            theme_prompt=self._prompt_var.get().strip()
            neg_prompt=DEFAULT_NEGATIVE_PROMPT if self._use_def_neg.get() else self._neg_var.get().strip()
            steps=self._steps_v.get(); strength=self._strength_v.get(); guidance=self._guidance_v.get()
            out_dir.mkdir(parents=True,exist_ok=True)

            ok=err=skip=0
            for i,src in enumerate(selected,1):
                if self._stop_ev.is_set():
                    self._log_safe("Parado pelo usuário.","warn"); break

                dst=out_dir/src.name
                self._log_safe(f"[{i}/{total}] {src.name}","info")

                if sem_ctx:
                    # ── txt2img puro: sem BLIP, sem imagem de entrada ────
                    word=random.choice(self._words) if self._words else "abstract"
                    self._log_safe(f"  Palavra sorteada: \"{word}\"","word")
                    status=process_txt2img(dst,word,steps,guidance,
                        log_fn=self._log_safe,stop_ev=self._stop_ev)
                else:
                    # ── img2img com BLIP ─────────────────────────────────
                    status=process_img2img(src,dst,theme_prompt,neg_prompt,
                        steps,strength,guidance,
                        log_fn=self._log_safe,stop_ev=self._stop_ev)

                if status=="ok":
                    ok+=1; self._log_safe(f"  ✓ {dst.name}","ok")
                elif status=="error": err+=1
                else: skip+=1

            self._log_safe("─"*52,"info")
            self._log_safe(f"Finalizado!  ✓ OK: {ok}   ✗ Erros: {err}   — Ignorados: {skip}","ok")
            self._log_safe(f"Replacements em: {out_dir}","ok")
            self._log_safe("PCSX2 → Config → Graphics → Texture Replacement → Enable ✓","warn")
        except Exception as e:
            self._log_safe(f"Erro inesperado: {e}","err")
        finally:
            self._running=False; self._stop_ev.clear()
            self.after(0,self._reset_btn)

    def _reset_btn(self):
        self._start_btn.config(text="INICIAR SWAPPER",bg=GREEN,activebackground=GREEN_DK)

if __name__=="__main__":
    App().mainloop()
