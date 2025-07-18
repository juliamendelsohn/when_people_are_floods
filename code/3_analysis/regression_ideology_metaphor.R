#!/usr/bin/env Rscript
library(optparse)
library(data.table)
library(lme4)
library(tidyr)
library(broom)
library(broom.mixed)
library(nnet)
library(forcats)
library(marginaleffects)
library(stargazer)
library(dplyr)
options("width"=200)


option_list = list(
  make_option("--includeFrames", action="store_true", default=FALSE, help="Include issue-generic frames as fixed effects")
)
opt = parse_args(OptionParser(option_list=option_list))
include_frames <- opt$includeFrames # TRUE or FALSE


concepts <- c('animal','commodity','parasite','pressure','vermin','war','water')
# In order to get columns that start with "score" and end with each concept
concept_cols <- c()
for (i in 1:length(concepts)) {
    concept_cols[i] <- paste('score',concepts[i],sep='_')
}

# Features to include in the regression model as controls (fixed effects)
features <- c('has_hashtag','has_mention','has_url','is_quote_status','is_reply',
                    'is_verified','log_chars','log_followers','log_following','log_statuses'
                    ,'year','month')


# If includeFrames is TRUE, these issue-generic frames will be incldued as fixed effects. 
frames <- c("Crime.and.Punishment","Cultural.Identity","Economic",
"Fairness.and.Equality","Health.and.Safety","Legality..Constitutionality..Jurisdiction",
"Morality.and.Ethics", "Policy.Prescription.and.Evaluation","Political.Factors.and.Implications",
"Security.and.Defense")

# If includeFrames is TRUE, we will create interaction terms between ideology and each frame
frames_interact <- c()
for (i in 1:length(frames)) {
    frames_interact[i] <- paste('ideology',frames[i],sep='*')
}

DATA_FILE <- "../../results/sample_data_with_metaphor_scores.tsv"
data <- read.csv(DATA_FILE,header=TRUE,sep='\t',quote="")

# Binarize ideology
data$magnitude_raw <- abs(data$ideology) # Create magnitude column for ideology
data$ideology <- ifelse(data$ideology > 0, 1, 0) # Sets ideology to 1 if > 0 (conservative), 0 otherwise (liberal)
# group centered mean magnitude
data <- data %>% group_by(ideology) %>% mutate(magnitude = magnitude_raw - mean(magnitude_raw)) %>% ungroup()
# z-score magnitude column
data[['magnitude']] <- scale(data[['magnitude']])[,1]


# Predict concept from ideology and frames
full_results = data.frame()
full_mfx_avg = data.frame()
full_mfx_group = data.frame()
models <- list()
print('starting regressions')

if (include_frames) {
    for (i in 1:length(concepts)) {
        print(concepts[i])
        df <- subset(data, select = c(concept_cols[i],'ideology','magnitude',frames,features))
        colnames(df)[1] <- "y"
        df<-df[complete.cases(df),]
        model <- lm(y ~ ideology:magnitude + . + (year/month) - year - month, data = df)
        print(summary(model))
        mfx_avg <- avg_slopes(model,variables=c('ideology','magnitude'))
        print(mfx_avg)
        mfx_group <- avg_slopes(model,by=c('ideology'),variables=c('magnitude'))
        models[[i]] <- model
        res <- tidy(model)
        res$concept <- concepts[i]
        full_results <- rbind(full_results,res)
        res_mfx_avg <- tidy(mfx_avg)
        res_mfx_avg$concept <- concepts[i]
        full_mfx_avg <- rbind(full_mfx_avg,res_mfx_avg)
        print(res_mfx_avg)
        res_mfx_group <- tidy(mfx_group)
        res_mfx_group$concept <- concepts[i]
        full_mfx_group <- rbind(full_mfx_group,res_mfx_group)
  }
} else {
    for (i in 1:length(concept_cols)) {
        df <- subset(data, select = c(concept_cols[i],'ideology','magnitude',features))
        colnames(df)[1] <- "y"
        df<-df[complete.cases(df),]
        model <- lm(y ~ ideology:magnitude + . + (year/month) - year - month, data = df)
        print(summary(model))
        mfx_avg <- avg_slopes(model,variables=c('ideology','magnitude'))
        mfx_group <- avg_slopes(model,by=c('ideology'),variables=c('magnitude'))
        models[[i]] <- model
        res <- tidy(model)
        res$concept <- concepts[i]
        full_results <- rbind(full_results,res)
        res_mfx_avg <- tidy(mfx_avg)
        res_mfx_avg$concept <- concepts[i]
        full_mfx_avg <- rbind(full_mfx_avg,res_mfx_avg)
        print(res_mfx_avg)
        res_mfx_group <- tidy(mfx_group)
        res_mfx_group$concept <- concepts[i]
        full_mfx_group <- rbind(full_mfx_group,res_mfx_group)
  }
} 

# Correct p values for "ideology" across all models
full_results['p.corrected'] <- full_results$p.value
ideology_orig_p <- full_results$p.value[full_results$term == "ideology"]
ideology_corrected_p <- p.adjust(ideology_orig_p, method = "holm", n = length(ideology_orig_p))
full_results$p.corrected[full_results$term == "ideology"] <- ideology_corrected_p


# Do the same for marginal effects
full_mfx_avg['p.corrected'] <- full_mfx_avg$p.value
ideology_orig_p <- full_mfx_avg$p.value[full_mfx_avg$term == "ideology"]
ideology_corrected_p <- p.adjust(ideology_orig_p, method = "holm", n = length(ideology_orig_p))
full_mfx_avg$p.corrected[full_mfx_avg$term == "ideology"] <- ideology_corrected_p


out.dir <- '../../results/analysis/'
dir.create(out.dir, showWarnings = FALSE)
out.file <- 'ideology_effects_on_metaphor'
print(out.file)

# # Set out.file name based on include_frames
if (include_frames) {
    out.file <- paste(out.file,'_frames',sep='')
} else {
    out.file <- paste(out.file,'_no_frames',sep='')
}

write.table(full_results,paste(out.dir,out.file,'_regression_results.tsv',sep=''),sep='\t')
write.table(full_mfx_avg,paste(out.dir,out.file,'_avg_marginal_effects.tsv',sep=''),sep='\t')
write.table(full_mfx_group,paste(out.dir,out.file,'_marginal_effects_by_ideology.tsv',sep=''),sep='\t')

#Get list of corrected p values for each model for stargazer
model_corrected_p_list <- list()
for (i in 1:length(concepts)) {
    model_corrected_p_list[[i]] <- full_results$p.corrected[full_results$concept == concepts[i]]
}

stargazer(models,out=paste(out.dir,out.file,'.tex',sep=''), 
        type='latex',no.space=TRUE,single.row=TRUE,star.cutoffs= c(0.05, 0.01, 0.001),
        report = "vc*",se=NULL,digits=3,column.labels=concepts,
        p.auto=FALSE,p=model_corrected_p_list)

stargazer(models,
        type='text',no.space=TRUE,single.row=TRUE,star.cutoffs= c(0.05, 0.01, 0.001),
        report = "vc*",se=NULL,digits=3,column.labels=concepts,
        p.auto=FALSE,p=model_corrected_p_list)



