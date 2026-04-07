(require 'package)

(setq package-archives
      '(("melpa" . "https://melpa.org/packages/")
        ("gnu" . "https://elpa.gnu.org/packages/")
        ("org" . "https://orgmode.org/elpa/")))

(package-initialize)

(unless (package-installed-p 'projectile)
  (package-refresh-contents)
  (package-install 'projectile))
(require 'projectile)
(projectile-mode +1)

;; installing lsp-ui for diagnostics display (squiggly lines, hover hints)
(unless (package-installed-p 'lsp-ui)
  (package-refresh-contents)
  (package-install 'lsp-ui))

(require 'lsp-ui)
(setq lsp-ui-doc-enable t)
(setq lsp-ui-doc-show-with-cursor t)
(setq lsp-ui-sideline-enable t)
(setq lsp-ui-sideline-show-diagnostics t)
(setq lsp-ui-sideline-show-hover t)

(unless package-archive-contents
  (package-refresh-contents))

(tool-bar-mode -1)
(menu-bar-mode -1)
(setq display-line-numbers-type 'relative)
(global-display-line-numbers-mode 1)

;;; Set location for external packages.
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(custom-enabled-themes '(gruber-darker))
 '(custom-safe-themes
   '("e13beeb34b932f309fb2c360a04a460821ca99fe58f69e65557d6c1b10ba18c7" default))
 '(inhibit-startup-screen t)
 '(package-selected-packages
   '(emmet-mode web-mode vue-mode all-the-icons python-black pyvenv lsp-pyright python-mode prettier-js typescript-mode js2-mode flycheck company lsp-mode gruber-darker-theme treemacs treemacs-projectile treemacs-all-the-icons)))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )

(setq ess-arg-function-offset 0)
;; (defun fontify-frame (frame)
;;   (set-frame-parameter frame 'font "Monospace-10"))

;; ;; Fontify current frame
;; (fontify-frame nil)
;; ;; Fontify any future frames
;; (push 'fontify-frame after-make-frame-functions)

;; dodajemy serwer, zeby mozna bylo uzywac trace z edit=TRUE
(require 'server)

;; udostepnienie elpy
;;(package-initialize)
;;(elpy-enable)

;; treemacs setup
(unless (package-installed-p 'treemacs)
  (package-refresh-contents)
  (package-install 'treemacs))

;; loading treemacs
(require 'treemacs)

;; configuring treemacs
(setq treemacs-position 'left
      treemacs-width 35
      treemacs-is-never-other-window t)

;; key binding to toggle treemacs
(global-set-key (kbd "C-c t") 'treemacs)

;; optional: adding project integration
(unless (package-installed-p 'treemacs-projectile)
  (package-refresh-contents)
  (package-install 'treemacs-projectile))
(with-eval-after-load 'treemacs
  (require 'treemacs-projectile))

;; Install the all-the-icons package first
(unless (package-installed-p 'all-the-icons)
  (package-refresh-contents)
  (package-install 'all-the-icons))


;; Modify your treemacs theme configuration
(with-eval-after-load 'treemacs
  (if (package-installed-p 'all-the-icons)
      (treemacs-load-theme "all-the-icons")
    (message "all-the-icons package is missing, using default theme")))

;; optional: adding nice icons (only if you're using a GUI version of Emacs)
(unless (package-installed-p 'treemacs-all-the-icons)
  (package-refresh-contents)
  (package-install 'treemacs-all-the-icons))
(treemacs-load-theme "all-the-icons")

;; remapowanie na polskie litery
(define-key key-translation-map (kbd "M-a") (kbd "ą"));
(define-key key-translation-map (kbd "M-A") (kbd "Ą"));

(define-key key-translation-map (kbd "M-c") (kbd "ć"));
(define-key key-translation-map (kbd "M-C") (kbd "Ć"));

(define-key key-translation-map (kbd "M-e") (kbd "ę"));
(define-key key-translation-map (kbd "M-E") (kbd "Ę"));

(define-key key-translation-map (kbd "M-l") (kbd "ł"));
(define-key key-translation-map (kbd "M-L") (kbd "Ł"));

(define-key key-translation-map (kbd "M-n") (kbd "ń"));
(define-key key-translation-map (kbd "M-N") (kbd "Ń"));

(define-key key-translation-map (kbd "M-n") (kbd "ó"));
(define-key key-translation-map (kbd "M-N") (kbd "Ó"));

(define-key key-translation-map (kbd "M-s") (kbd "ś"));
(define-key key-translation-map (kbd "M-S") (kbd "Ś"));

(define-key key-translation-map (kbd "M-`") (kbd "ź"));
(define-key key-translation-map (kbd "M-~") (kbd "Ź"));

(define-key key-translation-map (kbd "M-z") (kbd "ż"));
(define-key key-translation-map (kbd "M-Z") (kbd "Ż"));

(define-key key-translation-map (kbd "M-o") (kbd "ó"));
(define-key key-translation-map (kbd "M-O") (kbd "Ó"));

;;RUST part
;; adding package repositories if not already present
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)

;; installing rust-mode if not already installed
(unless (package-installed-p 'rust-mode)
  (package-refresh-contents)
  (package-install 'rust-mode))

;; enabling rust-mode
(require 'rust-mode)
(add-to-list 'auto-mode-alist '("\\.rs\\'" . rust-mode))

;; optional but recommended: adding rust-analyzer support
(unless (package-installed-p 'lsp-mode)
  (package-refresh-contents)
  (package-install 'lsp-mode))

;; configuring lsp-mode for rust
(require 'lsp-mode)
(add-hook 'rust-mode-hook 'lsp)
(add-to-list 'lsp-enabled-clients 'ts-ls) ;; For typescript-language-server
(add-hook 'js2-mode-hook #'lsp-deferred)

;; optional: adding company mode for completion
(unless (package-installed-p 'company)
  (package-install 'company))
(add-hook 'rust-mode-hook 'company-mode)

;; optional: adding flycheck for on-the-fly syntax checking
(unless (package-installed-p 'flycheck)
  (package-install 'flycheck))
(add-hook 'rust-mode-hook 'flycheck-mode)

;; optional: configuring format on save
(setq rust-format-on-save t)

(with-eval-after-load 'lsp-mode
  (add-to-list 'lsp-enabled-clients 'rust-analyzer))

;; Make sure rust-analyzer is installed
(unless (executable-find "rust-analyzer")
  (message "Installing rust-analyzer is recommended for Rust development"))

;; JAVASCRIPT
;; do this first:
;; ---
;; npm install -g typescript-language-server typescript
;; npm install -g prettier
;; ---

;; Install required packages
(unless (package-installed-p 'js2-mode)
  (package-refresh-contents)
  (package-install 'js2-mode))

(unless (package-installed-p 'typescript-mode)
  (package-install 'typescript-mode))

(unless (package-installed-p 'prettier-js)
  (package-install 'prettier-js))

;; Set global prettier args for semicolons
(setq prettier-js-args '("--semi"))

;; Configure js2-mode as the default for javascript files
(add-to-list 'auto-mode-alist '("\\.js\\'" . js2-mode))
(add-to-list 'auto-mode-alist '("\\.jsx\\'" . js2-mode))
(add-to-list 'auto-mode-alist '("\\.ts\\'" . typescript-mode))
(add-to-list 'auto-mode-alist '("\\.tsx\\'" . typescript-mode))


;; Hooks for JavaScript/TypeScript modes
(add-hook 'js2-mode-hook 
          (lambda ()
            (lsp-deferred)
            (prettier-js-mode)
            ;; Configure prettier to add semicolons
            (setq-local prettier-js-args '("--semi"))
            (company-mode)
            (flycheck-mode)))

(add-hook 'typescript-mode-hook 
          (lambda ()
            (lsp-deferred)
            (prettier-js-mode)
            ;; Configure prettier to add semicolons
            (setq-local prettier-js-args '("--semi"))
            (company-mode)
            (flycheck-mode)))

;; js2-mode specific settings
(setq js2-highlight-level 3)
(setq js2-mode-show-parse-errors t)
(setq js2-mode-show-strict-warnings t)
(setq js-indent-level 2)

;; PYTHON
;; need to do these first:
;; ------------
;; npm install -g pyright
;; pip install black
;; ------------
;; Create project-specific environments:

;; bashCopycd your_project
;; python -m venv .venv
;; source .venv/bin/activate
;; pip install black pyright

;; Use M-x pyvenv-activate in Emacs and select the .venv directory in your project when you work on it.
;; installing python development packages
(unless (package-installed-p 'python-mode)
  (package-refresh-contents)
  (package-install 'python-mode))

;; setting up LSP mode for Python
(unless (package-installed-p 'lsp-mode)
  (package-install 'lsp-mode))

;; adding lsp-pyright for better Python support
(unless (package-installed-p 'lsp-pyright)
  (package-install 'lsp-pyright))

;; automatically install pyright if not present
(unless (executable-find "pyright-langserver")
  (message "Installing pyright language server...")
  (start-process "npm" "*npm*" "npm" "install" "-g" "pyright"))

;; installing company mode for completion
(unless (package-installed-p 'company)
  (package-install 'company))

;; setting up flycheck for real-time syntax checking
(unless (package-installed-p 'flycheck)
  (package-install 'flycheck))

;; installing pyvenv for virtual environment support
(unless (package-installed-p 'pyvenv)
  (package-install 'pyvenv))

;; python mode hooks and configuration
(add-hook 'python-mode-hook #'lsp-deferred)
(add-hook 'python-mode-hook #'flycheck-mode)
(add-hook 'python-mode-hook #'company-mode)

;; configuring python indentation
(setq python-indent-offset 4)

;; setting up black for formatting
(unless (package-installed-p 'python-black)
  (package-install 'python-black))

;; configuring black with explicit path
;; Let black be found in the active virtual environment
(setq python-black-command (executable-find "black"))
(setq python-black-extra-args '("--line-length" "88"))

;; enable debug output for black
(setq python-black-debug t)

(add-hook 'python-mode-hook
          (lambda ()
            (python-black-on-save-mode 1)))

;; lsp performance tweaks
(setq gc-cons-threshold 100000000)
(setq read-process-output-max (* 1024 1024))
(setq lsp-idle-delay 0.5)

;; project detection for Python
(unless (package-installed-p 'projectile)
  (package-install 'projectile))

(projectile-mode +1)

;; suppress LSP client warning messages
(setq lsp-warn-no-matched-clients nil)

;; configuring lsp-mode specifically for python
(with-eval-after-load 'lsp-mode
  (setq lsp-enable-snippet nil)
  (setq lsp-pyright-multi-root nil)
  (setq lsp-pyright-auto-import-completions t)
  (setq lsp-pyright-auto-search-paths t)
  ;; explicitly enable only pyright
  (setq lsp-disabled-clients '(mspyls ruff semgrep-ls pylsp pyls copilot-ls))
  (add-to-list 'lsp-enabled-clients 'pyright))

;; virtual environment setup with proper variable declarations
(require 'pyvenv)
(defvar python-shell-virtualenv-path nil)
(defvar python-shell-virtualenv-root nil)
(pyvenv-mode 1)

;; automatically detect and activate virtualenvs
(setq pyvenv-workon (getenv "WORKON_HOME"))  ; virtualenvwrapper
(when (getenv "VIRTUAL_ENV")
  (pyvenv-activate (getenv "VIRTUAL_ENV")))

;; ESS - R
;;---------
;;C-c C-f formats the code (so it is NOT on save, like with the other languages
;;---------
;; installing ESS if not already installed
(unless (package-installed-p 'ess)
  (package-refresh-contents)
  (package-install 'ess))

;; loading ESS
(require 'ess-site)

;; configuring ESS for R
(setq ess-use-flymake nil)  ; disable flymake in favor of flycheck
(setq ess-style 'RStudio)   ; RStudio style formatting
(setq ess-eval-visibly t)   ; show evaluation results right away
(setq ess-ask-for-ess-directory nil)  ; don't ask for directory on startup
(setq ess-arg-function-offset 0)      ; your existing setting

;; setting up company mode for R completion
(unless (package-installed-p 'company)
  (package-install 'company))
(add-hook 'ess-r-mode-hook #'company-mode)

;; setting up flycheck for R
(unless (package-installed-p 'flycheck)
  (package-install 'flycheck))
(add-hook 'ess-r-mode-hook #'flycheck-mode)

;; format-on-save with formatR
(defun ess-format-region-or-buffer ()
  "Format the selected region or the entire buffer."
  (interactive)
  (let ((this-process (ess-get-process)))
    (when this-process
      (if (region-active-p)
          (let ((beg (region-beginning))
                (end (region-end)))
            (ess-format-region beg end))
        (ess-format-buffer)))))

(defun ess-format-buffer ()
  "Format the R buffer using formatR::tidy_source()."
  (interactive)
  (let ((proc (ess-get-process)))
    (when proc
      (let ((buf-content (buffer-string)))
        (with-current-buffer (process-buffer proc)
          (let ((formatted (ess-string-command 
                           (format "formatR::tidy_source(text = '%s', arrow = TRUE, width.cutoff = 80)\n" 
                                   (replace-regexp-in-string "'" "\\\\'" buf-content)))))
            (when formatted
              (with-current-buffer (current-buffer)
                (delete-region (point-min) (point-max))
                (insert formatted)))))))))

;; Add simple formatting commands that don't require R process
(defun ess-indent-region-or-buffer ()
  "Indent the selected region or the entire buffer."
  (interactive)
  (if (region-active-p)
      (indent-region (region-beginning) (region-end))
    (indent-region (point-min) (point-max))))

;; Add convenient key binding for formatting
(define-key ess-r-mode-map (kbd "C-c C-f") #'ess-indent-region-or-buffer)

;; Make sure TAB indents properly
(setq ess-tab-always-indent t)

;; additional ESS customizations
(setq ess-indent-with-fancy-comments nil)  ; don't indent comments differently
(setq ess-indent-level 2)                  ; standard R indentation
(setq ess-offset-arguments 'prev-line)     ; function arguments alignment
(setq ess-expression-offset 2)             ; expression indentation
(setq ess-nuke-trailing-whitespace-p t)    ; remove trailing whitespace

;; keybindings for pipe operator
(define-key ess-r-mode-map (kbd "C-S-m") " %>% ")
(define-key inferior-ess-r-mode-map (kbd "C-S-m") " %>% ")

;; stop creating ~ files
(setq make-backup-files nil)

;; VUE.JS
;; do this first:
;; ---
;; npm install -g @vue/language-server
;; npm install -g prettier prettier-plugin-vue
;; ---

;; Install required packages for Vue
(unless (package-installed-p 'vue-mode)
  (package-refresh-contents)
  (package-install 'vue-mode))

(unless (package-installed-p 'web-mode)
  (package-install 'web-mode))

;; Configure file associations for Vue
(add-to-list 'auto-mode-alist '("\\.vue\\'" . vue-mode))

;; Vue-specific settings
(setq vue-mode-packages
      '(vue-mode edit-indirect ssass-mode vue-html-mode))

;; Configure web-mode for better Vue template support
(with-eval-after-load 'web-mode
  (setq web-mode-markup-indent-offset 2)
  (setq web-mode-css-indent-offset 2)
  (setq web-mode-code-indent-offset 2)
  (setq web-mode-script-padding 2)
  (setq web-mode-style-padding 2))

;; Optional: Configure vue-mode to use web-mode for templates
(setq vue-mode-template-modes
      '(("html" . web-mode)
        ("jade" . jade-mode)
        ("pug" . pug-mode)))

;; LSP configuration for Vue
(with-eval-after-load 'lsp-mode
  ;; Register Vue language server
  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection "vue-language-server --stdio")
    :major-modes '(vue-mode)
    :server-id 'vue-ls))
  
  (add-to-list 'lsp-language-id-configuration '(vue-mode . "vue"))
  ;; Use vue-ls instead of vue-semantic-server
  (add-to-list 'lsp-enabled-clients 'vue-ls)

  ;; Allow LSP in non-project buffers
  (setq lsp-auto-guess-root t)
  (setq lsp-enable-file-watchers nil))

;; Vue mode hook - simplified and fixed
(add-hook 'vue-mode-hook
          (lambda ()
            ;; Disable ts-ls specifically for vue files
            (setq-local lsp-disabled-clients '(ts-ls))
            ;; Enable prettier-js-mode
            (prettier-js-mode)
            ;; Add format-on-save hook specifically for this buffer
            (add-hook 'before-save-hook 'prettier-js nil 'local)
            ;; Other modes
            (lsp-deferred)
            (company-mode)
            (flycheck-mode)))

;; AUTO-CLOSING HTML TAGS
;; Install emmet-mode for auto-closing tags and HTML shortcuts
(unless (package-installed-p 'emmet-mode)
  (package-refresh-contents)
  (package-install 'emmet-mode))

(require 'emmet-mode)

;; Enable emmet-mode in web-mode, vue-mode, and other markup modes
(add-hook 'web-mode-hook 'emmet-mode)
(add-hook 'vue-mode-hook 'emmet-mode)
(add-hook 'html-mode-hook 'emmet-mode)
(add-hook 'css-mode-hook 'emmet-mode)
(add-hook 'sgml-mode-hook 'emmet-mode)

;; Configure emmet settings
(setq emmet-move-cursor-between-quotes t)
(setq emmet-self-closing-tag-style " /")

;; Custom function to auto-close HTML tags like VSCode
(defun my/auto-close-html-tag ()
  "Automatically close HTML tag when '>' is typed, like VSCode."
  (interactive)
  (insert ">")
  (when (and (or (derived-mode-p 'web-mode)
                 (derived-mode-p 'vue-mode)
                 (derived-mode-p 'html-mode)
                 (derived-mode-p 'sgml-mode))
             (looking-back "<\\([a-zA-Z][a-zA-Z0-9-]*\\)\\([^>]*\\)>" (line-beginning-position)))
    (let ((tag-name (match-string 1)))
      ;; List of self-closing tags
      (unless (member tag-name '("img" "br" "hr" "input" "meta" "link" "area" "base" "col" "embed" "param" "source" "track" "wbr"))
        (save-excursion
          (insert (format "</%s>" tag-name)))))))

;; Bind the function to '>' key in relevant modes
(defun my/setup-auto-close-tag ()
  "Setup auto-close tag keybinding."
  (local-set-key (kbd ">") 'my/auto-close-html-tag))

(add-hook 'web-mode-hook 'my/setup-auto-close-tag)
(add-hook 'vue-mode-hook 'my/setup-auto-close-tag)
(add-hook 'html-mode-hook 'my/setup-auto-close-tag)
(add-hook 'sgml-mode-hook 'my/setup-auto-close-tag)
;; ELECTRIC PAIR
(electric-pair-mode -1)
;; Electric pairing off, but if want it for some reason, uncomment everything below
;; Keep electric-pair-mode for other characters


;; Enable electric-pair-mode globally for auto-pairing brackets, quotes, etc.
;; (electric-pair-mode 1)

;; Disable electric-pair in modes where we handle > manually
;; (add-hook 'web-mode-hook (lambda () (electric-pair-local-mode -1)))
;; (add-hook 'vue-mode-hook (lambda () (electric-pair-local-mode -1)))
;; (add-hook 'html-mode-hook (lambda () (electric-pair-local-mode -1)))
;; (add-hook 'sgml-mode-hook (lambda () (electric-pair-local-mode -1)))
