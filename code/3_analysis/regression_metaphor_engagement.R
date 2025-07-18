#!/usr/bin/env Rscript
library(optparse)
library(data.table)
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
  make_option("--includeFrames", action="store_true", default=FALSE, help="Include issue-generic frames as fixed effects"),
  make_option("--includeIdeology", action="store_true", default=FALSE, help="Include ideology as fixed effect")
)
opt = parse_args(OptionParser(option_list=option_list))


include_frames <- opt$includeFrames # TRUE or FALSE, default is FALSE
include_ideology <- opt$includeIdeology # TRUE or FALSE, default is FALSE


frames <- c("Crime.and.Punishment","Cultural.Identity","Economic",
            "Fairness.and.Equality","Health.and.Safety","Legality..Constitutionality..Jurisdiction",
            "Morality.and.Ethics", "Policy.Prescription.and.Evaluation","Political.Factors.and.Implications",
            "Security.and.Defense")                  
features <- c('has_hashtag','has_mention','has_url','is_quote_status','is_reply',
            'is_verified','log_chars','log_followers','log_following','log_statuses',
            'year','month')
outcomes <- c('log_favorites','log_retweets')
concepts <- c('animal','commodity','parasite','pressure','vermin','war','water')


create_outfile_prefix <- function(out.dir) {
    dir.create(out.dir, showWarnings = FALSE)
    out.file <- paste(out.dir,"metaphor_effects_on_engagement",sep='')
    if (include_frames) {
        out.file <- paste(out.file,'frames',sep='_')
    } else {
        out.file <- paste(out.file,'no_frames',sep='_')
    }
    if (include_ideology) {
        out.file <- paste(out.file,'with_ideology',sep='_')
    }
    return(out.file)
}

create_concept_cols <- function(concepts) {
    concept_cols <- c()
    for (i in 1:length(concepts)) {
        concept_cols[i] <- paste('score',concepts[i],sep='_')
    }
    return(concept_cols)
}

create_concept_interaction_cols <- function(concept_cols) {
    concept_interact_cols <- c()
    for (i in 1:length(concept_cols)) {
        concept_interact_cols[i] <- paste(concept_cols[i],':','ideology',sep='')
    }
    return(concept_interact_cols)
}

setup_data <- function(datasheet_file,concept_cols) {
    data <- read.csv(datasheet_file,header=TRUE,sep='\t',quote="")
    if (include_ideology) {
        data$magnitude_raw <- abs(data$ideology) # Create magnitude column for ideology
        data$ideology <- ifelse(data$ideology > 0, 1, 0) # Sets ideology to 1 if > 0 (conservative
        data <- data %>% group_by(ideology) %>% mutate(magnitude = magnitude_raw - mean(magnitude_raw)) %>% ungroup()
        data[['magnitude']] <- scale(data[['magnitude']])[,1]

    }
    data[concept_cols] <- scale(data[concept_cols])
    return(data)
}


DATA_FILE <- "../../results/sample_tweet_data_with_metaphor_scores.tsv"
out.dir <- '../../results/analysis/'
out.file <- create_outfile_prefix(out.dir)
concept_cols <- create_concept_cols(concepts)
data <- setup_data(DATA_FILE,concept_cols)


full_results = data.frame()
full_mfx_avg = data.frame()
full_mfx_group = data.frame()
models <- list()
print('starting regressions')

add_result <- function(full_results,model,outcome) {
    res <- tidy(model)
    res$outcome <- outcome
    full_results <- rbind(full_results,res)
    return(full_results)
}

# Create a function that runs the regression given an outcome column
construct_model <- function(outcome,include_frames,include_ideology) {
    cols_to_include <- c(outcome,concept_cols,features)
    if (include_frames) {cols_to_include <- c(cols_to_include,frames)}
    if (include_ideology) {cols_to_include <- c(cols_to_include,'ideology','magnitude')}
    df <- subset(data, select = cols_to_include)
    colnames(df)[1] <- "y"
    df<-df[complete.cases(df),]
    formula <- "y ~ . + (year/month) - year - month"
    if (include_ideology) {
        concept_interact_cols <- create_concept_interaction_cols(concept_cols)
        formula <- paste(formula,paste(concept_interact_cols,collapse=' + '),"ideology:magnitude",sep=' + ')
    }
    model <- lm(as.formula(formula), data = df)
    return(model)
}

for (i in 1:length(outcomes)) {
    outcome <- outcomes[i]
    model <- construct_model(outcome,include_frames,include_ideology)
    models[[i]] <- model
    full_results <- add_result(full_results,model,outcome)
    print(full_results)
    mfx_avg <- avg_slopes(model, variables=c(concept_cols))
    print(mfx_avg)
    full_mfx_avg <- add_result(full_mfx_avg,mfx_avg,outcome)
    if (include_ideology) {
        mfx_group <- avg_slopes(model,by=c('ideology'),variables=c(concept_cols))
        print(mfx_group)
        full_mfx_group <- add_result(full_mfx_group,mfx_group,outcome)
    }
}

correct_p_values <- function(result_table,concept_cols,outcomes,old_p_col_name) {
    variables <- c(concept_cols,outcomes)
    result_table[['p.corrected']] <- result_table[[old_p_col_name]]
    orig_p <- result_table[[old_p_col_name]][result_table$term %in% variables]
    print(length(orig_p))
    corrected_p <- p.adjust(orig_p, method = "holm", n = length(orig_p))
    result_table[['p.corrected']][result_table$term %in% variables] <- corrected_p
    return(result_table)
}


full_results <- correct_p_values(full_results,concept_cols,outcomes,'p.value')
print(full_results)
write.table(full_results,paste(out.file,'_regression_results.tsv',sep=''),sep='\t')


full_mfx_avg <- correct_p_values(full_mfx_avg,concept_cols,outcomes,'p.value')
write.table(full_mfx_avg,paste(out.file,'_avg_marginal_effects.tsv',sep=''),sep='\t')

if (include_ideology) {
    full_mfx_group <- correct_p_values(full_mfx_group,concept_cols,outcomes,'p.value')
    write.table(full_mfx_group,paste(out.file,'_marginal_effects_by_ideology.tsv',sep=''),sep='\t')

}

write_stargazer <- function(models,full_results,concept_cols,out.file){
    corrected_p_fav <- full_results[full_results$outcome == 'log_favorites',]$p.corrected
    corrected_p_rt <- full_results[full_results$outcome == 'log_retweets',]$p.corrected
    corrected_p <- list(corrected_p_fav,corrected_p_rt)
    print(length(models))
    stargazer(models,out=out.file, 
        type='latex',no.space=TRUE,single.row=TRUE,star.cutoffs= c(0.05, 0.01, 0.001),
        report = "vc*",se=NULL,digits=3,column.labels=c('favorites','retweets'),
        p.auto=FALSE,p=corrected_p)
    stargazer(models,
        type='text',no.space=TRUE,single.row=TRUE,star.cutoffs= c(0.05, 0.01, 0.001),
        report = "vc*",se=NULL,digits=3,column.labels=c('favorites','retweets'),
        p.auto=FALSE,p=corrected_p)

}

write_stargazer(models,full_results,concept_cols,paste(out.file,'_engagement.tex',sep=''))
